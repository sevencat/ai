using SixLabors.ImageSharp;

namespace sevencat.ai.yolo;

public class Sam3ResultItem
{
	public float score { get; set; }
	public RectangleF box { get; set; }
	public float[] mask { get; set; }
}

public class Sam3Result
{
	public int mask_model_width { get; set; }
	public int mask_model_height { get; set; }

	public List<Sam3ResultItem> items { get; set; } = [];
}