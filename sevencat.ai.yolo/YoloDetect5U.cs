using sevencat.ai.yolo.entity;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace sevencat.ai.yolo;

public class YoloDetect5U : YoloDetect11
{
	public YoloDetect5U(YoloConfiguration config, byte[] modeldata) : base(config, modeldata)
	{
	}

}