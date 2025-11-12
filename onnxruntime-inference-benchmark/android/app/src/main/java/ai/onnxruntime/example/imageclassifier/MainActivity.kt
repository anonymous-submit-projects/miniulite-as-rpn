// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

package ai.onnxruntime.example.imageclassifier

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.example.imageclassifier.databinding.ActivityMainBinding
import android.content.Intent
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
import kotlin.system.measureTimeMillis

class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding
    private val scope = CoroutineScope(Job() + Dispatchers.Main)
    private var ortEnv: OrtEnvironment? = null

    private val TAG = "ONNX_BENCHMARK"
    // Define the dimensions that your model expects.
    private val MODEL_INPUT_WIDTH = 512
    private val MODEL_INPUT_HEIGHT = 512
    private val TOTAL_RUNS = 10
    private val WARMUP_RUNS = 5
    private val STANDARD_MODEL = R.raw.atten_unet_inp512

    override fun onCreate(savedInstanceState: Bundle?) {
        Log.d(TAG, "OnCreate Started")
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)

        setupInfoTextView()

        setContentView(binding.root)
        ortEnv = OrtEnvironment.getEnvironment()

        // Configure the RecyclerView.
        binding.recyclerViewResults.layoutManager = LinearLayoutManager(this)

        binding.buttonStartBenchmark.setOnClickListener {
            binding.buttonStartBenchmark.isEnabled = false // Disable the button during testing.
            binding.textViewAvgTime.text = "Testing...."
            binding.recyclerViewResults.adapter = null // Clears previous results

            // Start the benchmark in a coroutine to avoid freezing the UI.
            scope.launch {
                runBenchmark()
            }
        }

        binding.buttonGotoWnet.setOnClickListener {
            val intent = Intent(this, WNetActivity::class.java)
            startActivity(intent)
        }
    }

    private fun setupInfoTextView() {
        // Get the resource name for each model.
        // This will return the filename without the extension, e.g., "mini_ulite_inp512"
        val standardModelName = resources.getResourceEntryName(STANDARD_MODEL)
        // Assemble the complete information string.
        val infoText = """
        Configure the models to be used in the FIRST_MODEL and SECOND_MODEL variables; if necessary, also configure PATCH_DIM according to the SECOND_MODEL.

        Actual configurations:
        STANDARD_MODEL = $standardModelName
        MODEL_INPUT_WIDTH = $MODEL_INPUT_WIDTH
        MODEL_INPUT_HEIGHT = $MODEL_INPUT_HEIGHT
        TOTAL_RUNS = $TOTAL_RUNS
        WARMUP_RUNS = $WARMUP_RUNS
    """.trimIndent()

        binding.infoBenchmark.text = infoText
    }

    // Load the ONNX model from the res/raw folder.
    private suspend fun readModel(): ByteArray = withContext(Dispatchers.IO) {
        // Use the name of your model that you put in the raw folder.
        resources.openRawResource(STANDARD_MODEL).readBytes()
    }

    private suspend fun createOrtSession(): OrtSession? = withContext(Dispatchers.Default) {
        ortEnv?.createSession(readModel())
    }

    // Load the 16 images from the assets folder.
    private suspend fun loadImages(): List<Bitmap> = withContext(Dispatchers.IO) {
        val imageList = mutableListOf<Bitmap>()
        // Assuming your images are named from 0001.png to 0016.png
        Log.d(TAG, "Loading images...")
        for (i in 1..16) {
            val fileName = "${String.format("%04d", i)}.png" // ex: 0001.png
            assets.open(fileName).use { inputStream ->
                imageList.add(BitmapFactory.decodeStream(inputStream))
            }
        }
        Log.d(TAG, "Images loaded.")
        return@withContext imageList
    }

    // Converts a bitmap to the tensor format that the model expects (FloatBuffer).
    private fun preprocess(bitmap: Bitmap): FloatBuffer {
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT, true)
        // The FloatBuffer must have the same total size.
        val floatBuffer = FloatBuffer.allocate(MODEL_INPUT_WIDTH * MODEL_INPUT_HEIGHT * 3)
        floatBuffer.rewind()

        val pixels = IntArray(MODEL_INPUT_WIDTH * MODEL_INPUT_HEIGHT)
        resizedBitmap.getPixels(pixels, 0, resizedBitmap.width, 0, 0, resizedBitmap.width, resizedBitmap.height)

        // Your dataloader uses .convert("RGB"), so the order is RGB.
        // The normalization transformation in PyTorch for [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        // is equivalent to (value/255.0 - 0.5) / 0.5, which simplifies to (value/255.0 * 2) - 1.
        // Your normalization is mathematically correct.

        // The key is to fill the buffer in the order C, H, W (Channel, Height, Width).
        for (y in 0 until MODEL_INPUT_HEIGHT) {
            for (x in 0 until MODEL_INPUT_WIDTH) {
                val pixelValue = pixels[y * MODEL_INPUT_WIDTH + x]

                // Extract and normalize the canals, just like you were doing before.
                val r = (((pixelValue shr 16) and 0xFF) / 255.0f) * 2.0f - 1.0f
                val g = (((pixelValue shr 8) and 0xFF) / 255.0f) * 2.0f - 1.0f
                val b = ((pixelValue and 0xFF) / 255.0f) * 2.0f - 1.0f

                // Calculates the correct index in the buffer for the layout (C, H, W)
                val rIndex = y * MODEL_INPUT_WIDTH + x
                val gIndex = rIndex + (MODEL_INPUT_WIDTH * MODEL_INPUT_HEIGHT)
                val bIndex = gIndex + (MODEL_INPUT_WIDTH * MODEL_INPUT_HEIGHT)

                // Place the values in their respective channel "plans".
                floatBuffer.put(rIndex, r)
                floatBuffer.put(gIndex, g)
                floatBuffer.put(bIndex, b)
            }
        }

        floatBuffer.rewind()
        return floatBuffer
    }

    // Converts the output tensor of the segmentation into a Bitmap.
    private fun postprocess(outputTensor: OnnxTensor): Bitmap {
        // The expected output shape is [batch, 1, height, width] or [batch, num_classes, height, width]
        val outputBuffer = outputTensor.floatBuffer
        val shape = outputTensor.info.shape
        val height = shape[2].toInt()
        val width = shape[3].toInt()
        val numClasses = shape[1].toInt()

        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val pixels = IntArray(width * height)

        if (numClasses == 1) { // Binary Mask
            for (i in 0 until width * height) {
                val score = outputBuffer.get(i)
                // If score > 0 (after sigmoid), paint with one color, otherwise, transparent.
                pixels[i] = if (score > 0) Color.argb(254, 255, 255, 255) else Color.argb(254, 0, 0, 0)
            }
        } else { // Multiple classes
            // Generates random colors for each class.
            val random = Random()
            val colors = IntArray(numClasses) {
                Color.argb(128, random.nextInt(256), random.nextInt(256), random.nextInt(256))
            }

            for (y in 0 until height) {
                for (x in 0 until width) {
                    var maxClassIndex = 0
                    var maxClassScore = -Float.MAX_VALUE
                    // Find the class with the highest score for this pixel.
                    for (c in 0 until numClasses) {
                        val score = outputBuffer.get(c * (width * height) + y * width + x)
                        if (score > maxClassScore) {
                            maxClassScore = score
                            maxClassIndex = c
                        }
                    }
                    // Paint the pixel with the color of the winning class (or transparent for class 0/background).
                    pixels[y * width + x] = if (maxClassIndex > 0) colors[maxClassIndex] else Color.TRANSPARENT
                }
            }
        }
        bitmap.setPixels(pixels, 0, width, 0, 0, width, height)
        return bitmap
    }

    data class BenchmarkResultMain(    val totalBenchmarkTime: Long,
                                       val totalInferenceTime: Long,
                                       val benchmarkExecutionCount: Int,
                                       val resultsList: List<SegmentationResult>
    )

    private suspend fun runBenchmark() {
        Log.d(TAG, "Initializing benchmark...")
        binding.textViewAvgTime.text = "Loading model..."
        val ortSession = createOrtSession() ?: return
        Log.d(TAG, "Model loaded.")

        binding.textViewAvgTime.text = "Loading images..."
        val images = loadImages()
        delay(500)

        binding.textViewAvgTime.text = "Executing inferences with $WARMUP_RUNS Warm-up rounds..."
        delay(1000)

        val inputName = ortSession.inputNames.first()

        val benchmarkResultDeferred = scope.async(Dispatchers.Default) {
            var totalInferenceTime = 0L
            val localResultsList = mutableListOf<SegmentationResult>()
            var benchmarkExecutionCount = 0

            val totalBenchmarkTime = measureTimeMillis {
                repeat(TOTAL_RUNS + WARMUP_RUNS) { runIndex ->
                    val phase = if (runIndex < WARMUP_RUNS) "WARMUP" else "BENCHMARK"
                    Log.d(TAG, "Executing round ${runIndex + 1} ($phase)...")

                    for (imageToTest in images) {
                        val preprocessedInput = preprocess(imageToTest)
                        val inputTensor = OnnxTensor.createTensor(
                            ortEnv,
                            preprocessedInput,
                            longArrayOf(1, 3, MODEL_INPUT_HEIGHT.toLong(), MODEL_INPUT_WIDTH.toLong())
                        )

                        var sessionResult: OrtSession.Result? = null
                        var outputTensor: OnnxTensor? = null

                        val inferenceTime = measureTimeMillis {
                            sessionResult = ortSession.run(mapOf(inputName to inputTensor))

                            // 2. We obtain the tensor from the result.
                            val outputName = ortSession.outputNames.first()
                            outputTensor = sessionResult?.get(outputName)?.get() as OnnxTensor
                        }

                        if (runIndex >= WARMUP_RUNS) {
                            totalInferenceTime += inferenceTime
                            benchmarkExecutionCount++
                        }

                        if (runIndex == (TOTAL_RUNS + WARMUP_RUNS - 1)) {
                            // Now 'outputTensor' is open and can be used safely.
                            val maskBitmap = postprocess(outputTensor!!)
                            localResultsList.add(SegmentationResult(imageToTest, maskBitmap))
                        }

                        // --- Manual Memory Management ---
                        // We close everything up after use.
                        inputTensor.close()
                        outputTensor?.close()
                        sessionResult?.close()
                    }
                }
            }
            ortSession.close()

            BenchmarkResultMain(
                totalBenchmarkTime = totalBenchmarkTime,
                totalInferenceTime = totalInferenceTime,
                benchmarkExecutionCount = benchmarkExecutionCount,
                resultsList = localResultsList
            )
        }

        val result = benchmarkResultDeferred.await()

        val averageTime = if (result.benchmarkExecutionCount > 0) result.totalInferenceTime.toDouble() / result.benchmarkExecutionCount else 0.0
        val totalInferenceTimeInSeconds = result.totalInferenceTime / 1000.0
        val fps = if (totalInferenceTimeInSeconds > 0) result.benchmarkExecutionCount / totalInferenceTimeInSeconds else 0.0

        val summaryText = "FPS (inference): %.2f\n".format(fps) +
                "Avg time (inference): %.2f ms\n".format(averageTime) +
                "Total: %d benchmark inferences in %.2f s".format(
                    result.benchmarkExecutionCount,
                    result.totalBenchmarkTime / 1000.0
                )
        binding.textViewAvgTime.text = summaryText

        binding.recyclerViewResults.adapter = ResultAdapter(result.resultsList)
        binding.buttonStartBenchmark.isEnabled = true
        Log.d(TAG, "Benchmark finished. ${result.benchmarkExecutionCount} benchmark inferences were performed.")
    }




    override fun onDestroy() {
        super.onDestroy()
        ortEnv?.close()
    }
}
