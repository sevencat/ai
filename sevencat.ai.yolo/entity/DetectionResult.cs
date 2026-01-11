using SixLabors.ImageSharp;

namespace sevencat.ai.yolo.entity;

public class DetectionResultItem
{
	public RectangleF Bounds { get; set; }

	public float Confidence { get; set; }

	public YoloName Name { get; set; }

	public float Angle { get; set; }

	public int NameIndex { get; set; }
	public int Index { get; set; }

	public int CompareTo(DetectionResultItem other) => Confidence.CompareTo(other.Confidence);
}