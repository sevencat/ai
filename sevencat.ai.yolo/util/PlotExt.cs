using sevencat.ai.yolo.entity;
using SixLabors.Fonts;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Drawing.Processing;
using SixLabors.ImageSharp.Processing;

namespace sevencat.ai.yolo.util;

public static class PlotExt
{
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