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

		//只有这个地方需要锁！！！
		using var results = _session.Run(new List<NamedOnnxValue>
		{
			NamedOnnxValue.CreateFromTensor("images", input0)
		});

		//把数据复制出来，因为这个数据有可能是在gpu,不在cpu!!!
		var output0 = results[0].AsTensor<float>().ToDenseTensor();
		var otensor = new MemoryTensor<float>(output0.Buffer, [.. output0.Dimensions]);
		{
			List<RawBoundingBox> rawBoundingBoxes = new List<RawBoundingBox>();
			var boxStride = otensor.Strides[1];
			var boxesCount = otensor.Dimensions[2];
			var namesCount = _metadata.Names.Length;

			var tensorSpan = otensor.Buffer.Span;
			for (var boxIndex = 0; boxIndex < boxesCount; boxIndex++)
			{
				for (var nameIndex = 0; nameIndex < namesCount; nameIndex++)
				{
					var confidence = tensorSpan[(nameIndex + 4) * boxStride + boxIndex];

					if (confidence <= _config.Confidence)
					{
						continue;
					}

					ParseBox(tensorSpan, boxStride, boxIndex, out var bounds, out var angle);

					if (bounds.Width == 0 || bounds.Height == 0)
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
			var imageAdjustment = new ImageAdjustmentHelper(_metadata);

			var size = image.Size;
			var adjustment = imageAdjustment.Calculate(size);

			var result = new List<DetectionResultItem>();
			foreach (var box in boxes)
			{
				result.Add(new DetectionResultItem
				{
					Name = _metadata.Names[box.NameIndex],
					Bounds = imageAdjustment.Adjust(box.Bounds, adjustment),
					Confidence = box.Confidence,
				});
			}

			return result;
		}
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