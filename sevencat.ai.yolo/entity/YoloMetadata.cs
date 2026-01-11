using Microsoft.ML.OnnxRuntime;
using SixLabors.ImageSharp;

namespace sevencat.ai.yolo.entity;

public class YoloMetadata
{
	public string Author { get; set; }

	public string Description { get; set; }

	public string Version { get; set; }

	public int BatchSize { get; set; }

	public Size ImageSize { get; set; }

	public string Task { get; set; }

	public YoloName[] Names { get; set; }

	public void ParseFrom(ModelMetadata mm)
	{
		var custmetamap = mm.CustomMetadataMap;
		Author = custmetamap["author"];
		Description = custmetamap["description"];
		Version = custmetamap["version"];
		Task = custmetamap["task"];
		BatchSize = int.Parse(custmetamap["batch"]);
		ImageSize = ParseSize(custmetamap["imgsz"]);
		Names = ParseNames(custmetamap["names"]);
	}
	
	
	#region Parsers

	private static Size ParseSize(string text)
	{
		text = text[1..^1]; // '[640, 640]' => '640, 640'

		var split = text.Split(", ");

		var y = int.Parse(split[0]);
		var x = int.Parse(split[1]);

		return new Size(x, y);
	}

	private static YoloName[] ParseNames(string text)
	{
		text = text[1..^1];

		var split = text.Split(", ");
		var count = split.Length;

		var names = new YoloName[count];

		for (int i = 0; i < count; i++)
		{
			var value = split[i];

			var valueSplit = value.Split(": ");

			var id = int.Parse(valueSplit[0]);
			var name = valueSplit[1][1..^1].Replace('_', ' ');

			names[id] = new YoloName(id, name);
		}

		return names;
	}

	#endregion
}