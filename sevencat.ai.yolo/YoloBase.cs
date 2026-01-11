using Microsoft.ML.OnnxRuntime;
using sevencat.ai.yolo.entity;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace sevencat.ai.yolo;

public abstract class YoloBase
{
	protected InferenceSession _session;
	protected YoloConfiguration _config;
	protected YoloMetadata _metadata;

	protected YoloBase(YoloConfiguration config, byte[] modeldata)
	{
		_config = config;
		var sessionopt = CreateSessionOptions(config);
		_session = new InferenceSession(modeldata, sessionopt);
		_metadata = new YoloMetadata();
		_metadata.ParseFrom(_session.ModelMetadata);
	}


	protected SessionOptions CreateSessionOptions(YoloConfiguration config)
	{
		if (config.UseCuda)
			return SessionOptions.MakeSessionOptionWithCudaProvider(config.CudaDeviceId);
		return new SessionOptions();
	}

	
}