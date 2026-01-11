using sevencat.ai.yolo;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using YoloConfiguration = sevencat.ai.yolo.YoloConfiguration;

namespace test;

class Program
{
	private const string mdlpath = @"F:\ai\models\yolo";
	private const string imgpath = @"F:\ai\image\test";

	static void Main(string[] args)
	{
		var outputdir = AppDomain.CurrentDomain.BaseDirectory;

		var model = Path.Combine(mdlpath, "yolov11s.onnx");
		var imgfile = Path.Combine(imgpath, "sports.jpg");

		var modeldat = File.ReadAllBytes(model);
		var config = new YoloConfiguration();
		config.Confidence = 0.3f;
		config.UseCuda = false;
		var yolo = new YoloDetect11(config, modeldat);
		var img = Image.Load(imgfile);
		using var img2 = img.CloneAs<Rgb24>();
		var result = yolo.Detect(img2);
		return;
	}
}