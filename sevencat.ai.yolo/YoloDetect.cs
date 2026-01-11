using sevencat.ai.yolo.entity;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace sevencat.ai.yolo;

public abstract class YoloDetect : YoloBase
{
	protected YoloDetect(YoloConfiguration config, byte[] modeldata) : base(config, modeldata)
	{
	}

	public abstract List<DetectionResultItem> Detect(Image<Rgb24> image);
}