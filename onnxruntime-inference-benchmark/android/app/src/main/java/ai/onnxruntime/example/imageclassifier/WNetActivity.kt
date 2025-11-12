package ai.onnxruntime.example.imageclassifier

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.example.imageclassifier.databinding.ActivityWnetBinding
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import androidx.recyclerview.widget.LinearLayoutManager
import kotlinx.coroutines.*
import java.nio.FloatBuffer
import java.util.*
import kotlin.math.max
import kotlin.math.min
import kotlin.system.measureTimeMillis

// Data class for storing patch coordinates.
data class PatchCoords(
    val batchIndex: Int,
    val yStart: Int,
    val xStart: Int,
    val yEnd: Int,
    val xEnd: Int
)

class WNetActivity : AppCompatActivity() {

    private lateinit var binding: ActivityWnetBinding
    // CoroutineScope linked to the Activity lifecycle to prevent memory leaks.
    private val activityScope = CoroutineScope(Dispatchers.Main + Job())

    private var ortEnv: OrtEnvironment? = null

    // Sessions for both models
    private var model1Session: OrtSession? = null
    private var model2Session: OrtSession? = null

    // --- CONTROL VARIABLES ---
    private val NUM_IMAGES_TO_LOAD = 16
    private val TOTAL_RUNS = 10
    private val WARMUP_RUNS = 5
    private val MODEL_INPUT_WIDTH = 512
    private val MODEL_INPUT_HEIGHT = 512
    private val FIRST_MODEL = R.raw.mini_ulite_inp512
    private val PATCH_DIM = 384
    private val SECOND_MODEL = R.raw.atten_unet_inp384
    private val TAG = "ONNX_BENCHMARK"

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityWnetBinding.inflate(layoutInflater)
        setContentView(binding.root)

        setupInfoTextView()

        binding.recyclerViewWnetResults.layoutManager = LinearLayoutManager(this)

        // Initializes the background models as soon as the screen is created.
        initializeModels()

        binding.buttonStartWnetBenchmark.setOnClickListener {
            binding.buttonStartWnetBenchmark.isEnabled = false
            binding.textViewTotalTime.text = "Initializing test..."
            binding.recyclerViewWnetResults.adapter = null

            activityScope.launch {
                runWNetBenchmark()
            }
        }
    }

    private fun setupInfoTextView() {
        // Get the resource name for each model.
        // This will return the filename without the extension, e.g., "mini_ulite_inp512"
        val firstModelName = resources.getResourceEntryName(FIRST_MODEL)
        val secondModelName = resources.getResourceEntryName(SECOND_MODEL)

        val infoText = """
        Configure the models to be used in the FIRST_MODEL and SECOND_MODEL variables; if necessary, also configure PATCH_DIM according to the SECOND_MODEL.

        Actual configurations:
        FIRST_MODEL = $firstModelName
        MODEL_INPUT_WIDTH = $MODEL_INPUT_WIDTH
        MODEL_INPUT_HEIGHT = $MODEL_INPUT_HEIGHT
        SECOND_MODEL = $secondModelName
        PATCH_DIM = $PATCH_DIM
        TOTAL_RUNS = $TOTAL_RUNS
        WARMUP_RUNS = $WARMUP_RUNS
        NUM_IMAGES_TO_LOAD = $NUM_IMAGES_TO_LOAD
    """.trimIndent()

        binding.infoPipelineBenchmark.text = infoText
    }

    // Function to initialize models in the background, called once.
    private fun initializeModels() {
        activityScope.launch(Dispatchers.Default) { // Roda em uma thread de background
            Log.d(TAG, "Initializing models in background...")
            try {
                ortEnv = OrtEnvironment.getEnvironment()
                val model1Bytes = resources.openRawResource(FIRST_MODEL).readBytes()
                val model2Bytes = resources.openRawResource(SECOND_MODEL).readBytes()
                model1Session = ortEnv?.createSession(model1Bytes)
                model2Session = ortEnv?.createSession(model2Bytes)
                withContext(Dispatchers.Main) {
                    Log.d(TAG, "Models initializeds with success.")
                }
            } catch (e: Exception) {
                Log.e(TAG, "Failed to load models", e)
            }
        }
    }

    private suspend fun runWNetBenchmark() {
        // Check if the models are ready.
        if (model1Session == null || model2Session == null) {
            binding.textViewTotalTime.text = "The models are not ready yet. Please try again."
            binding.buttonStartWnetBenchmark.isEnabled = true
            return
        }

        binding.textViewTotalTime.text = "Loading $NUM_IMAGES_TO_LOAD image(s)..."
        val images = loadImages()
        delay(1000)

        binding.textViewTotalTime.text = "Performing inference, including $WARMUP_RUNS warmup runs..."
        delay(1000)

        // The async function immediately returns a 'Deferred' (a promise of a result).
        // The UI thread does not get blocked waiting.
        val benchmarkResultDeferred = activityScope.async(Dispatchers.Default) {
            // All the computationally intensive code goes in here.

            var totalPipelineTime = 0L
            var totalInference1Time = 0L
            var totalExtractionTime = 0L
            var totalInference2Time = 0L
            var totalReconstructionTime = 0L
            val localResultsList = mutableListOf<SegmentationResult>()

            val totalBenchmarkTime = measureTimeMillis {
                repeat(TOTAL_RUNS + WARMUP_RUNS) { runIndex ->
                    images.forEachIndexed { imageIndex, imageBitmap ->

                        val model1InputBuffer = preprocess(imageBitmap, MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT)
                        val model1InputName = model1Session!!.inputNames.first()
                        val tensor1 = OnnxTensor.createTensor(ortEnv, model1InputBuffer, longArrayOf(1, 3, MODEL_INPUT_HEIGHT.toLong(), MODEL_INPUT_WIDTH.toLong()))

                        var inference1Time: Long
                        var extractionTime: Long
                        var inference2Time: Long
                        var reconstructionTime: Long
                        var finalMaskBitmap: Bitmap

                        // --- MEDIÇÃO DO PIPELINE ---
                        val pipelineTime = measureTimeMillis {
                            val model1Result: OrtSession.Result
                            inference1Time = measureTimeMillis {
                                model1Result = model1Session!!.run(mapOf(model1InputName to tensor1))
                            }
                            val model1OutputTensor = model1Result.get(0) as OnnxTensor
                            val patchesCoords = findPatchesFromModel1Output(model1OutputTensor, imageIndex)

                            if (patchesCoords.isNotEmpty()) {
                                val batchPatchesTensor: OnnxTensor
                                extractionTime = measureTimeMillis {
                                    val patchTensors = mutableListOf<FloatBuffer>()
                                    for (coords in patchesCoords) {
                                        val patchInputBuffer = extractPatchFromBuffer(model1InputBuffer, MODEL_INPUT_WIDTH, coords, PATCH_DIM)
                                        patchTensors.add(patchInputBuffer)
                                    }
                                    batchPatchesTensor = createBatchTensor(patchTensors)
                                }

                                val model2Result: OrtSession.Result
                                val model2InputName = model2Session!!.inputNames.first()
                                inference2Time = measureTimeMillis {
                                    model2Result = model2Session!!.run(mapOf(model2InputName to batchPatchesTensor))
                                }
                                val model2OutputTensor = model2Result.get(0) as OnnxTensor

                                val reconstructedMask: OnnxTensor
                                reconstructionTime = measureTimeMillis {
                                    reconstructedMask = reconstructOutput(model2OutputTensor, patchesCoords)
                                }
                                finalMaskBitmap = postprocess(reconstructedMask)

                                batchPatchesTensor.close()
                                model2OutputTensor.close()
                                model2Result.close()
                            } else {
                                extractionTime = 0
                                inference2Time = 0
                                reconstructionTime = 0
                                finalMaskBitmap = Bitmap.createBitmap(MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT, Bitmap.Config.ARGB_8888).apply { eraseColor(Color.TRANSPARENT) }
                            }

                            tensor1.close()
                            model1Result.close()
                        } // --- FIM DA MEDIÇÃO DO PIPELINE ---

                        // *** LÓGICA DE WARMUP ***
                        if (runIndex >= WARMUP_RUNS) {
                            totalPipelineTime += pipelineTime
                            totalInference1Time += inference1Time
                            totalExtractionTime += extractionTime
                            totalInference2Time += inference2Time
                            totalReconstructionTime += reconstructionTime
                        }

                        if (runIndex == (TOTAL_RUNS + WARMUP_RUNS - 1)) {
                            localResultsList.add(SegmentationResult(imageBitmap, finalMaskBitmap))
                        }

                        if (imageIndex == 0) {
                            val phase = if (runIndex < WARMUP_RUNS) "WARMUP" else "BENCHMARK"
                            Log.d(TAG, "--- ROUND ${runIndex + 1} ($phase) ---")
                            Log.d(TAG, "Inference 1: $inference1Time ms")
                            Log.d(TAG, "Extraction/Batch: $extractionTime ms")
                            Log.d(TAG, "Inference 2: $inference2Time ms")
                            Log.d(TAG, "Reconstruct: $reconstructionTime ms")
                            Log.d(TAG, "---------------------------------")
                        }
                    }
                }
            }

            // Returns an object containing all the results needed for the UI.
            // This is the value that 'await()' will receive.
            BenchmarkResult(
                totalBenchmarkTime = totalBenchmarkTime,
                totalPipelineTime = totalPipelineTime,
                totalInference1Time = totalInference1Time,
                totalExtractionTime = totalExtractionTime,
                totalInference2Time = totalInference2Time,
                totalReconstructionTime = totalReconstructionTime,
                resultsList = localResultsList,
                totalImages = images.size
            )
        }

        // --- Safe UI Update ---
        // 'await()' suspends the coroutine (without blocking the thread) until the result...
        // Once the 'async' block is ready, the code below executes on the Main Thread.
        val result = benchmarkResultDeferred.await()

        // Now, use the 'result' object to calculate and update the UI.
        val totalExecutions = result.totalImages * TOTAL_RUNS
        val averageTime = if (totalExecutions > 0) result.totalPipelineTime.toDouble() / totalExecutions else 0.0
        val totalPipelineTimeInSeconds = result.totalPipelineTime / 1000.0
        val fps = if (totalPipelineTimeInSeconds > 0) totalExecutions / totalPipelineTimeInSeconds else 0.0
        val totalBenchmarkTimeInSeconds = result.totalBenchmarkTime / 1000.0

        if (TOTAL_RUNS > 1) {
            Log.d(TAG, "--- FINAL AVG (Benchmark: ${totalExecutions} runs) ---")
            Log.d(TAG, "Avg Inference 1: ${result.totalInference1Time.toDouble() / totalExecutions} ms")
            Log.d(TAG, "Avg Extraction/Batch: ${result.totalExtractionTime.toDouble() / totalExecutions} ms")
            Log.d(TAG, "Avg Inference 2: ${result.totalInference2Time.toDouble() / totalExecutions} ms")
            Log.d(TAG, "Avg Reconstruct: ${result.totalReconstructionTime.toDouble() / totalExecutions} ms")
            Log.d(TAG, "------------------------------------")
        }

        val summaryText = "FPS (Pipeline): %.2f\n".format(fps) +
                "Avg Time (Pipeline): %.2f ms\n".format(averageTime) +
                "Total: %d runs in %.2f s".format(totalExecutions, totalBenchmarkTimeInSeconds)

        binding.textViewTotalTime.text = summaryText
        binding.recyclerViewWnetResults.adapter = ResultAdapter(result.resultsList)
        binding.buttonStartWnetBenchmark.isEnabled = true
    }

    // Create this data class (it can be at the end of the file, outside the WNetActivity class).
    // to encapsulate the benchmark results and keep the code clean.
    data class BenchmarkResult(
        val totalBenchmarkTime: Long,
        val totalPipelineTime: Long,
        val totalInference1Time: Long,
        val totalExtractionTime: Long,
        val totalInference2Time: Long,
        val totalReconstructionTime: Long,
        val resultsList: List<SegmentationResult>,
        val totalImages: Int
    )






    private fun reconstructOutput(batchOutput: OnnxTensor, coordsList: List<PatchCoords>): OnnxTensor {
        val outputShape = batchOutput.info.shape
        val numClasses = outputShape[1].toInt()
        val finalOutputBuffer = FloatBuffer.allocate(1 * numClasses * MODEL_INPUT_HEIGHT * MODEL_INPUT_WIDTH)
        val finalOutputArray = finalOutputBuffer.array() // Retrieves the underlying array for direct access.

        val sourceBuffer = batchOutput.floatBuffer
        val sourcePatchPlaneSize = PATCH_DIM * PATCH_DIM
        val destPlaneSize = MODEL_INPUT_WIDTH * MODEL_INPUT_HEIGHT

        // Temporary array to copy an entire line from the patch at once.
        val rowData = FloatArray(PATCH_DIM)

        // For each patch processed in the batch
        for (i in coordsList.indices) {
            val coords = coordsList[i]
            val sourcePatchOffset = i * (numClasses * sourcePatchPlaneSize)

            // For each output channel of the patch
            for (c in 0 until numClasses) {
                val sourceChannelOffset = sourcePatchOffset + c * sourcePatchPlaneSize
                val destChannelOffset = c * destPlaneSize

                // For each line (y) within the patch
                for (y in 0 until PATCH_DIM) {
                    // Line start position in the source buffer (of the patch)
                    val sourceRowStartPos = sourceChannelOffset + y * PATCH_DIM
                    sourceBuffer.position(sourceRowStartPos)
                    // Block copy: reads the entire line from the patch into the temporary array.
                    sourceBuffer.get(rowData)

                    // Destination position of the row in the final array (of the 512x512 image)
                    val destY = coords.yStart + y
                    // It ensures that we will not write outside the vertical limits.
                    if (destY < MODEL_INPUT_HEIGHT) {
                        val destRowStartPos = destChannelOffset + destY * MODEL_INPUT_WIDTH + coords.xStart
                        // Bulk copy: writes the temporary array to the destination array.
                        // System.arraycopy is extremely fast for this.
                        System.arraycopy(rowData, 0, finalOutputArray, destRowStartPos, PATCH_DIM)
                    }
                }
            }
        }

        // The finalOutputBuffer is now populated by its underlying array.
        // There's no need to use rewind(), as the Tensor will be created from the entire buffer.
        return OnnxTensor.createTensor(ortEnv, finalOutputBuffer, longArrayOf(1, numClasses.toLong(), MODEL_INPUT_HEIGHT.toLong(), MODEL_INPUT_WIDTH.toLong()))
    }


    private suspend fun loadImages(): List<Bitmap> = withContext(Dispatchers.IO) {
        val imageList = mutableListOf<Bitmap>()
        for (i in 1..NUM_IMAGES_TO_LOAD) { // Use the control variable
            val fileName = "${String.format("%04d", i)}.png"
            assets.open(fileName).use { inputStream ->
                imageList.add(BitmapFactory.decodeStream(inputStream))
            }
        }
        return@withContext imageList
    }

    private fun preprocess(bitmap: Bitmap, width: Int, height: Int): FloatBuffer {
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, width, height, true)
        val floatBuffer = FloatBuffer.allocate(width * height * 3)
        // There's no need to use `rewind()` here, because a `put` statement with an absolute index doesn't advance the position.

        val pixels = IntArray(width * height)
        resizedBitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        // The normalization to [-1, 1] and the RGB order are already correct.
        // The key point is filling the buffer in the format (Channel, Height, Width) - CHW.

        val planeSize = width * height
        for (y in 0 until height) {
            for (x in 0 until width) {
                val pixelValue = pixels[y * width + x]

                // Extracts and normalizes the color channels for the range [-1, 1]
                val r = (((pixelValue shr 16) and 0xFF) / 255.0f) * 2.0f - 1.0f
                val g = (((pixelValue shr 8) and 0xFF) / 255.0f) * 2.0f - 1.0f
                val b = ((pixelValue and 0xFF) / 255.0f) * 2.0f - 1.0f

                // Calculates the index of the current pixel (position H, W)
                val pixelIndex = y * width + x

                // Place each channel value in its respective memory "plane":
                // Plan R (first third of the buffer)
                floatBuffer.put(pixelIndex, r)
                // Plan G (second third of the buffer)
                floatBuffer.put(pixelIndex + planeSize, g)
                // Plan B (third third of the buffer)
                floatBuffer.put(pixelIndex + (2 * planeSize), b)
            }
        }

        floatBuffer.rewind() // Ensures the buffer position is at the beginning before returning.
        return floatBuffer
    }

    /**
     * Extracts a patch from a pre-processed FloatBuffer.
     *
     * @param sourceBuffer The FloatBuffer of the entire image (512x512, CHW format).
     * @param sourceWidth The width of the source image (512).
     * @param coords The coordinates (yStart, xStart) and size of the patch to be extracted.
     * @param patchDim The patch dimension (256).
     * @return A new FloatBuffer containing only the patch data (256x256, CHW format).
     */
    private fun extractPatchFromBuffer(
        sourceBuffer: FloatBuffer,
        sourceWidth: Int,
        coords: PatchCoords,
        patchDim: Int
    ): FloatBuffer {
        val patchFloatBuffer = FloatBuffer.allocate(patchDim * patchDim * 3)
        val sourcePlaneSize = sourceWidth * sourceWidth // 512 * 512

        for (c in 0 until 3) { // For each channel (R, G, B)
            val sourceChannelOffset = c * sourcePlaneSize
            val destChannelOffset = c * (patchDim * patchDim)

            for (y in 0 until patchDim) { // For each line of the patch
                val sourceY = coords.yStart + y
                val sourceRowOffset = sourceChannelOffset + sourceY * sourceWidth + coords.xStart

                // Prepares a temporary array for the patch line.
                val rowData = FloatArray(patchDim)
                // Get the current buffer position to restore later.
                val originalPosition = sourceBuffer.position()
                // Move the buffer pointer to the beginning of the line you want to copy.
                sourceBuffer.position(sourceRowOffset)
                // Copies the entire line from the source buffer to the temporary array.
                sourceBuffer.get(rowData)
                // Restores the pointer to its original position.
                sourceBuffer.position(originalPosition)

                // Now, copy the data from the temporary array to the destination buffer.
                val destRowOffset = destChannelOffset + y * patchDim
                patchFloatBuffer.position(destRowOffset)
                patchFloatBuffer.put(rowData)
            }
        }

        patchFloatBuffer.rewind()
        return patchFloatBuffer
    }


    private fun findPatchesFromModel1Output(outputTensor: OnnxTensor, batchIndex: Int): List<PatchCoords> {
        val shape = outputTensor.info.shape
        val h = shape[2].toInt()
        val w = shape[3].toInt()
        val outputBuffer = outputTensor.floatBuffer

        val indexes = mutableListOf<Pair<Int, Int>>()

        for (i in 0 until h * w) {
            val rawValue = outputBuffer.get(i)
            val score = (1.0f / (1.0f + kotlin.math.exp(-rawValue)))
            if (score > 0.5) {
                val y = i / w
                val x = i % w
                indexes.add(Pair(y, x))
            }
        }

        if (indexes.isEmpty()) {
            return emptyList()
        }

        var sumY = 0.0
        var sumX = 0.0
        var minY = h
        var maxY = 0
        var minX = w
        var maxX = 0

        for (point in indexes) {
            sumY += point.first
            sumX += point.second
            minY = min(minY, point.first)
            maxY = max(maxY, point.first)
            minX = min(minX, point.second)
            maxX = max(maxX, point.second)
        }

        val yCenter = sumY / indexes.size
        val xCenter = sumX / indexes.size

        val patchesCoords = mutableListOf<PatchCoords>()
        val patchDim = PATCH_DIM

        val yStart = max(0, min(h - patchDim, (yCenter - patchDim / 2).toInt()))
        val xStart = max(0, min(w - patchDim, (xCenter - patchDim / 2).toInt()))
        // --- CORREÇÃO: Usar o 'batchIndex' recebido para criar o PatchCoords ---
        patchesCoords.add(PatchCoords(batchIndex, yStart, xStart, yStart + patchDim, xStart + patchDim))

        if (maxY - minY + 1 > patchDim) {
            for (y in minY..maxY step patchDim) {
                if (y == yStart) continue
                val yS = max(0, min(h - patchDim, y))
                patchesCoords.add(PatchCoords(batchIndex, yS, xStart, yS + patchDim, xStart + patchDim))
            }
        }

        if (maxX - minX + 1 > patchDim) {
            for (x in minX..maxX step patchDim) {
                if (x == xStart) continue
                val xS = max(0, min(w - patchDim, x))
                patchesCoords.add(PatchCoords(batchIndex, yStart, xS, yStart + patchDim, xS + patchDim))
            }
        }

        return patchesCoords.distinct()
    }

    private fun cropPatch(originalBitmap: Bitmap, coords: PatchCoords): Bitmap {
        return Bitmap.createBitmap(originalBitmap, coords.xStart, coords.yStart, PATCH_DIM, PATCH_DIM)
    }

    private fun createBatchTensor(patchTensors: List<FloatBuffer>): OnnxTensor {
        if (patchTensors.isEmpty()) {
            throw IllegalArgumentException("The patch tensor list cannot be empty.")
        }
        val numPatches = patchTensors.size
        val patchSize = 3 * PATCH_DIM * PATCH_DIM
        val batchBuffer = FloatBuffer.allocate(numPatches * patchSize)
        patchTensors.forEach { patchBuffer ->
            patchBuffer.rewind()
            batchBuffer.put(patchBuffer)
        }
        batchBuffer.rewind()
        val batchShape = longArrayOf(numPatches.toLong(), 3, PATCH_DIM.toLong(), PATCH_DIM.toLong())
        return OnnxTensor.createTensor(ortEnv, batchBuffer, batchShape)
    }

    private fun postprocess(outputTensor: OnnxTensor): Bitmap {
        val outputBuffer = outputTensor.floatBuffer
        val shape = outputTensor.info.shape
        val height = shape[2].toInt()
        val width = shape[3].toInt()
        val numClasses = shape[1].toInt()
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val pixels = IntArray(width * height)

        if (numClasses == 1) { // Binary Mask
            for (i in 0 until width * height) {
                val rawValue = outputBuffer.get(i)
                val score = (1.0f / (1.0f + kotlin.math.exp(-rawValue))) // Sigmoid
                pixels[i] = if (score > 0.5) Color.argb(254, 255, 255, 255) else Color.argb(254, 0, 0, 0)
            }
        } else { // Multiple classes
            val random = Random()
            val colors = IntArray(numClasses) {
                Color.argb(128, random.nextInt(256), random.nextInt(256), random.nextInt(256))
            }
            for (y in 0 until height) {
                for (x in 0 until width) {
                    var maxClassIndex = 0
                    var maxClassScore = -Float.MAX_VALUE
                    for (c in 0 until numClasses) {
                        val score = outputBuffer.get(c * (width * height) + y * width + x)
                        if (score > maxClassScore) {
                            maxClassScore = score
                            maxClassIndex = c
                        }
                    }
                    pixels[y * width + x] = if (maxClassIndex > 0) colors[maxClassIndex] else Color.TRANSPARENT
                }
            }
        }
        bitmap.setPixels(pixels, 0, width, 0, 0, width, height)
        return bitmap
    }

    override fun onDestroy() {
        super.onDestroy()
        // Cancels all coroutines started in this scope when the activity is destroyed.
        activityScope.cancel()
        model1Session?.close()
        model2Session?.close()
    }
}
