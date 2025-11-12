package ai.onnxruntime.example.imageclassifier

import android.graphics.Bitmap
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import androidx.recyclerview.widget.RecyclerView

// Classe de dados para um par de resultados
data class SegmentationResult(val originalImage: Bitmap, val maskImage: Bitmap)

class ResultAdapter(private val results: List<SegmentationResult>) :
    RecyclerView.Adapter<ResultAdapter.ViewHolder>() {

    class ViewHolder(view: View) : RecyclerView.ViewHolder(view) {
        val originalImageView: ImageView = view.findViewById(R.id.image_view_original)
        val maskImageView: ImageView = view.findViewById(R.id.image_view_mask)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.list_item_result, parent, false)
        return ViewHolder(view)
    }

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        val result = results[position]
        holder.originalImageView.setImageBitmap(result.originalImage)
        holder.maskImageView.setImageBitmap(result.maskImage)
    }

    override fun getItemCount() = results.size
}
