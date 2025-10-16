package com.example.chexpert

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import org.pytorch.torchvision.TensorImageUtils
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

class PyTorchModelLoader (){
    private var model: Module? = null
    private var inputSize: Int = 224

    constructor(inputSize: Int) : this() {
        this.inputSize = inputSize
    }

    companion object {
//        private const val INPUT_SIZE = 224 // Adjust based on your model

        // Normalization values (ImageNet standard - adjust for your model)
        private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val STD = floatArrayOf(0.229f, 0.224f, 0.225f)
    }

    /**
     * Load model from assets folder
     */
    fun loadModelFromAssets(context: Context, modelFileName: String) {
        try {
            val modelPath = assetFilePath(context, modelFileName)
            model = Module.load(modelPath)
        } catch (e: IOException) {
            e.printStackTrace()
        }
    }

    /**
     * Load model from file path
     */
    fun loadModelFromPath(modelPath: String) {
        model = Module.load(modelPath)
    }

    /**
     * Run inference on a bitmap image
     */
    fun predict(bitmap: Bitmap): FloatArray {
        // Resize bitmap to model input size
        val resizedBitmap = Bitmap.createScaledBitmap(
            bitmap, inputSize, inputSize, true
        )

        // Convert bitmap to tensor
        val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
            resizedBitmap,
            MEAN,
            STD
        )

        // Run inference
        val outputTensor = model?.forward(IValue.from(inputTensor))?.toTensor()
            ?: throw IllegalStateException("Model not loaded")

        // Get output as float array
        val output = outputTensor.dataAsFloatArray

        // Apply sigmoid to get probabilities
        return applySigmoid(output)
    }

    /**
     * Apply sigmoid function: 1 / (1 + e^(-x))
     */
    private fun applySigmoid(values: FloatArray): FloatArray {
        return FloatArray(values.size) { i ->
            (1.0 / (1.0 + Math.exp(-values[i].toDouble()))).toFloat()
        }
    }

    /**
     * Load image from assets
     */
    fun loadImageFromAssets(context: Context, imageFileName: String): Bitmap? {
        return try {
            context.assets.open(imageFileName).use { inputStream ->
                BitmapFactory.decodeStream(inputStream)
            }
        } catch (e: IOException) {
            e.printStackTrace()
            null
        }
    }

    /**
     * Helper method to copy asset file to internal storage
     */
    @Throws(IOException::class)
    private fun assetFilePath(context: Context, assetName: String): String {
        val file = File(context.filesDir, assetName)

        if (file.exists() && file.length() > 0) {
            return file.absolutePath
        }

        context.assets.open(assetName).use { inputStream ->
            FileOutputStream(file).use { outputStream ->
                val buffer = ByteArray(4 * 1024)
                var read: Int
                while (inputStream.read(buffer).also { read = it } != -1) {
                    outputStream.write(buffer, 0, read)
                }
                outputStream.flush()
            }
        }

        return file.absolutePath
    }

    /**
     * Example usage
     */
    fun example(context: Context) {
        // Load model
        loadModelFromAssets(context, "model.ptl")

        // Load and process image
        val image = loadImageFromAssets(context, "test_image.jpg") ?: return

        // Run inference
        val output = predict(image)

        // Process output (e.g., get top prediction)
        val maxIndex = output.indices.maxByOrNull { output[it] } ?: 0
        val maxValue = output[maxIndex]

        println("Predicted class: $maxIndex")
        println("Confidence: $maxValue")
    }
}