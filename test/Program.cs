using sevencat.ai.entity;
using sevencat.ai.openai;
using sevencat.ai.sam;
using sevencat.ai.util;
using sevencat.ai.yolo;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
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

	static void TestOpenAi1()
	{
		var aiclient = AiChatUtil.CreateOpenAiClient("sk-lm-agwPtBE7:7nvYrCX055awyV7VwE4y", "http://127.0.0.1:1234/v1");
		var cc = aiclient.GetChat("lfm2.5-vl-450m");
		var cp = new ChatPromptBase();
		cp.SysPrompt = "你是一名安全巡检人员.";
		cp.UserPrompt =
			"请用json返回图片中下列内容：1、是否确定有人抽烟（mwsmoke_exist),以及相关描述(mwsmoke_desc)，2、是否确定有火(fire_exist)以及描述(fire_desc)，3、是否确定有浓烟(smoke_exist)以及描述(smoke_desc),4、是否确定有不带头盔的人（mnhelm_exist)以及描述(mnhelm_desc),";

		//{"fire":"火","smoke":"烟","mnhelm":"不带头盔"}
		cp.Tags = new Dictionary<string, string>()
		{
			{ "fire", "火" },
			{ "smoke", "烟" },
			{ "mnhelm", "不带头盔" },
		};
		using var img = Image.Load("d:\\safedoor.jpg");
		var ret = cc.Analyze(img,cp).Result;
		return;
	}
	
	static void TestOpenAi2()
	{
		var aiclient = AiChatUtil.CreateOpenAiClient("sk-lm-agwPtBE7:7nvYrCX055awyV7VwE4y", "http://127.0.0.1:1234/v1");
		var cc = aiclient.GetChat("qwen/qwen3-vl-4b");
		var cp = new ChatPromptBase();
		cp.SysPrompt = "你是一名安全巡检人员.";
		cp.UserPrompt =
			"请用json返回图片中下列内容：比较这两张图， 是否有同一部车停在同样的位置（carstay_exist),以及相关描述(carstay_desc)";

		//{"fire":"火","smoke":"烟","mnhelm":"不带头盔"}
		cp.Tags = new Dictionary<string, string>()
		{
			{ "carstay", "有车停留" }
		};
		using var img1 = Image.Load("d:\\safedoor.jpg");
		using var img2 = Image.Load("d:\\safedoor.jpg");
		var ret = cc.Analyze(img1,img2,cp).Result;
		return;
	}

	static void Main(string[] args)
	{
		TestYolo();
	}
}