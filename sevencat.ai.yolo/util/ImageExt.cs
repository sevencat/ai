using sevencat.ai.yolo.entity;
using SixLabors.Fonts;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Drawing.Processing;
using SixLabors.ImageSharp.Formats.Jpeg;
using SixLabors.ImageSharp.Formats.Png;
using SixLabors.ImageSharp.Processing;

namespace sevencat.ai.yolo.util;

public static class ImageExt
{
	//JpegEncoder encoder
	public static string ToBase64JpgUrl(this Image img, JpegEncoder encoder)
	{
		var binjpegimg = img.ToJpegImage(encoder);
		var imgurl = "data:image/jpeg;base64," + Convert.ToBase64String(binjpegimg);
		return imgurl;
	}

	public static byte[] ToJpegImage(this Image img, JpegEncoder encoder)
	{
		using (var ms = new MemoryStream())
		{
			img.SaveAsJpeg(ms, encoder);
			return ms.ToArray();
		}
	}

	public static async Task<string> ToBase64JpgUrl(this Image img)
	{
		var binjpegimg = await img.ToJpegImage();
		var imgurl = "data:image/jpeg;base64," + Convert.ToBase64String(binjpegimg);
		return imgurl;
	}

	public static string ToBase64PngUrl(this Image img)
	{
		return img.ToBase64String(PngFormat.Instance);
	}

	public static async Task<byte[]> ToPngImage(this Image img)
	{
		using (var ms = new MemoryStream())
		{
			await img.SaveAsPngAsync(ms);
			return ms.ToArray();
		}
	}

	public static async Task<byte[]> ToJpegImage(this Image img)
	{
		using (var ms = new MemoryStream())
		{
			await img.SaveAsJpegAsync(ms);
			return ms.ToArray();
		}
	}

	public static void PlotImage(this Image img, List<DetectionResultItem> bbitems, string fontname = "Arial")
	{
		// 预定义一组颜色用于不同类别
		var colors = new Color[]
		{
			Color.Red, Color.Green, Color.Blue, Color.Orange,
			Color.Purple, Color.Cyan, Color.Magenta, Color.Yellow,
			Color.Lime, Color.Teal, Color.HotPink, Color.Gold
		};

		var fontFamily = SystemFonts.Get(fontname);
		var font = fontFamily.CreateFont(12f);

		img.Mutate(ctx =>
		{
			foreach (var item in bbitems)
			{
				var color = colors[item.Name.Id % colors.Length];
				var pen = Pens.Solid(color, 2f);
				var bounds = item.Bounds;

				// 绘制边界框
				ctx.Draw(pen, bounds);

				// 绘制标签
				var label = $"{item.Name.Name} {item.Confidence:P1}";
				var textSize = TextMeasurer.MeasureSize(label, new TextOptions(font));

				var labelX = bounds.X;
				var labelY = bounds.Y - textSize.Height - 4;
				if (labelY < 0) labelY = bounds.Y;

				var labelRect = new RectangleF(labelX, labelY, textSize.Width + 6, textSize.Height + 4);

				// 绘制标签背景
				ctx.Fill(color, labelRect);

				// 绘制标签文字
				ctx.DrawText(label, font, Color.White, new PointF(labelX + 3, labelY + 2));
			}
		});
	}
}