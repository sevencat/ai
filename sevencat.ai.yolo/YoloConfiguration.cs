namespace sevencat.ai.yolo;

public class YoloConfiguration
{
	public bool UseCuda { get; set; } = true;

	public int CudaDeviceId { get; set; } = 0;

	/// <summary>
	/// Specify the minimum confidence value for including a result. Default is 0.3f.
	/// </summary>
	public float Confidence { get; set; } = .3f;

	/// <summary>
	/// Specify the minimum IoU value for Non-Maximum Suppression (NMS). Default is 0.45f.
	/// </summary>
	public float IoU { get; set; } = .45f;
}