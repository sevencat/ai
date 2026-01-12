namespace sevencat.ai.yolo;

public class SamConfiguration
{
	public bool UseCuda { get; set; } = true;

	public int CudaDeviceId { get; set; } = 0;

	/// <summary>
	/// Specify the minimum confidence value for including a result. Default is 0.3f.
	/// </summary>
	public float Confidence { get; set; } = .3f;

	public string VisionEncoderModelPath { get; set; }
	public string TextEncoderModelPath { get; set; }
	public string GeometryEncoderModelPath { get; set; }
	public string DecoderEncoderModelPath { get; set; }
	public string TokenizerPath { get; set; }

	public void SetModelPath(string rootpath)
	{
		VisionEncoderModelPath = Path.Combine(rootpath, "vision-encoder-fp16.onnx");
		TextEncoderModelPath = Path.Combine(rootpath, "text-encoder-fp16.onnx");
		GeometryEncoderModelPath = Path.Combine(rootpath, "geometry-encoder-fp16.onnx");
		DecoderEncoderModelPath = Path.Combine(rootpath, "decoder-fp16.onnx");
		TokenizerPath = Path.Combine(rootpath, "tokenizer.json");
	}
}