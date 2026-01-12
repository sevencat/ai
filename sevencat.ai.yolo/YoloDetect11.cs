using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using sevencat.ai.yolo.entity;
using sevencat.ai.yolo.util;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace sevencat.ai.yolo;

public class YoloDetect11 : YoloDetect
{
	public YoloDetect11(YoloConfiguration config, byte[] modeldata) : base(config, modeldata)
	{
	}

	public List<RawBoundingBox> ParseRawBoundBox(Span<float> tensorSpan)
	{
		List<RawBoundingBox> rawBoundingBoxes = new List<RawBoundingBox>();
		var outputmeta = _session.OutputMetadata.First().Value;
		rawBoundingBoxes.Capacity = outputmeta.Dimensions[1] * outputmeta.Dimensions[2];

		var boxStride = outputmeta.Dimensions[2];
		var boxesCount = outputmeta.Dimensions[2];
		var namesCount = _metadata.Names.Length;

		for (var boxIndex = 0; boxIndex < boxesCount; boxIndex++)
		{
			//这里先解析这个box!!!
			var x = tensorSpan[0 * boxStride + boxIndex];
			var y = tensorSpan[1 * boxStride + boxIndex];
			var w = tensorSpan[2 * boxStride + boxIndex];
			var h = tensorSpan[3 * boxStride + boxIndex];

			var bounds = new RectangleF(x - w / 2, y - h / 2, w, h);
			if (bounds.Width == 0 || bounds.Height == 0)
			{
				continue;
			}

			var angle = float.NegativeZero;

			//再识别每个nameindex，后面80个8400每个都是置信度
			for (var nameIndex = 0; nameIndex < namesCount; nameIndex++)
			{
				var confidence = tensorSpan[(nameIndex + 4) * boxStride + boxIndex];
				if (confidence <= _config.Confidence)
				{
					continue;
				}

				RawBoundingBox curitem = new RawBoundingBox
				{
					Index = boxIndex,
					NameIndex = nameIndex,
					Confidence = confidence,
					Bounds = bounds,
					Angle = angle
				};
				rawBoundingBoxes.Add(curitem);
			}
		}

		var boxspan = CollectionsMarshal.AsSpan(rawBoundingBoxes);
		var boxes = NonMaxSuppressionUtil.Apply(boxspan, _config.IoU);
		return boxes.ToList();
	}

	public override List<DetectionResultItem> Detect(Image<Rgb24> image)
	{
		// Resize the input image
		using var resized = ResizeImage(image, out var padding);

		var inputmetadata = _session.InputMetadata["images"];
		var inputlen = inputmetadata.GetShapeLen();
		var inputdata = new float[inputlen];
		var input0 = new DenseTensor<float>(inputdata, inputmetadata.Dimensions);

		var target = new MemoryTensor<float>(inputdata, inputmetadata.Dimensions);
		ImageUtil.NormalizerPixelsToTensor(resized, target, padding);

		var outputmetadata = _session.OutputMetadata["output0"];
		var outputlen = outputmetadata.GetShapeLen();
		var outputdat = new float[outputlen];
		var output0 = new DenseTensor<float>(outputdat, outputmetadata.Dimensions);
		//只有这个地方需要锁！！！
		_session.Run([NamedOnnxValue.CreateFromTensor("images", input0)],
			[NamedOnnxValue.CreateFromTensor("output0", output0)]
		);

		var boxes = ParseRawBoundBox(new Span<float>(outputdat));
		var imageAdjustment = new ImageAdjustmentHelper(_metadata);
		var adjustment = imageAdjustment.Calculate(image.Size);

		var result = boxes.Select(x => new DetectionResultItem()
		{
			Name = _metadata.Names[x.NameIndex],
			Bounds = imageAdjustment.Adjust(x.Bounds, adjustment),
			Confidence = x.Confidence,
		}).ToList();
		return result;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	protected virtual void ParseBox(Span<float> tensor, int boxStride, int boxIndex, out RectangleF bounds,
		out float angle)
	{
		var x = tensor[0 + boxIndex];
		var y = tensor[1 * boxStride + boxIndex];
		var w = tensor[2 * boxStride + boxIndex];
		var h = tensor[3 * boxStride + boxIndex];

		bounds = new RectangleF(x - w / 2, y - h / 2, w, h);

		angle = float.NegativeZero;
	}

	private Image<Rgb24> ResizeImage(Image<Rgb24> image, out Vector<int> padding)
	{
		// Get the model image input size
		var inputSize = _metadata.ImageSize;

		// Create resize options
		var options = new ResizeOptions()
		{
			Size = inputSize,

			// Select resize mode according to 'KeepAspectRatio'
			Mode = ResizeMode.Max,

			// Select faster resampling algorithm
			Sampler = KnownResamplers.NearestNeighbor
		};

		// Create resized image
		var resized = image.Clone(x => x.Resize(options));

		// Calculate padding,这个是在两边留
		padding =
		(
			(inputSize.Width - resized.Size.Width) / 2,
			(inputSize.Height - resized.Size.Height) / 2
		);

		// Return the resized image
		return resized;
	}
}