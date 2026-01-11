using Microsoft.ML.OnnxRuntime;

namespace sevencat.ai.yolo.util;

public static class OnnxExt
{
	public static int GetShapeLen(this NodeMetadata metadata)
	{
		var result = 1;
		foreach (var dim in metadata.Dimensions)
			result = result * dim;
		return result;
	}
}