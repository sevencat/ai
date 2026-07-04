using sevencat.ai.yolo;
using sevencat.ai.yolo.entity;
using sevencat.ai.yolo.util;
using SixLabors.Fonts;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Drawing.Processing;
using SixLabors.ImageSharp.Processing;
using YoloConfiguration = sevencat.ai.yolo.YoloConfiguration;

namespace test;

class Program
{
	private const string mdlpath = @"F:\ai\models\yolo";
	private const string imgpath = @"F:\ai\image\test";

	static void TestYolo()
	{
		var outputdir = AppDomain.CurrentDomain.BaseDirectory;

		var model = Path.Combine(mdlpath, "yolo11n.onnx");
		var imgfile = Path.Combine(imgpath, "sports.jpg");

		var modeldat = File.ReadAllBytes(model);
		var config = new YoloConfiguration();
		config.Confidence = 0.3f;
		config.UseCuda = false;
		config.InterOpNumThreads = 1;
		config.IntraOpNumThreads = 1;
		var yolo = new YoloDetect11(config, modeldat);
		using var img = Image.Load(imgfile);
		using var img2 = img.CloneAs<Rgb24>();
		var result = yolo.Detect(img2);
		img2.PlotImage(result);
		File.Delete("d:\\yolo.jpg");
		img2.SaveAsJpeg("d:\\yolo.jpg");
		return;
	}


	static void TestSam3()
	{
		Configuration.Default.PreferContiguousImageBuffers = true;
		var modeldir = @"F:\ai\models\sam3";
		var cfg = new SamConfiguration();
		cfg.UseCuda = false;
		cfg.SetModelPath(modeldir);
		var sam3 = new Sam3Infer(cfg);


		var imagePath = Path.Combine(modeldir, "x7.png");

		var img = Image.Load(imagePath);
		var session = new Sam3Session();
		session.org_image = img.CloneAs<Rgb24>();
		session.org_image_height = img.Height;
		session.org_image_width = img.Width;
		session.prompt_text = "person";

		var result = sam3.Handle(session);
		session.PlotOrgImage(result);
		var outputdir = AppDomain.CurrentDomain.BaseDirectory;
		session.org_image.SaveAsJpeg(Path.Combine(outputdir, "ttt.jpg"));
	}

	static void Main(string[] args)
	{
		TestYolo();
	}
}