using Microsoft.ML.OnnxRuntime;

namespace sevencat.ai.yolo.util;

public static class OnnxExt
{
	public static int GetShapeLen(this NodeMetadata metadata, int defdim = 1)
	{
		var result = 1;
		foreach (var dim in metadata.Dimensions)
		{
			var curdim = dim;
			if (curdim <= 0)
				curdim = defdim;
			result = result * curdim;
		}

		return result;
	}

	//这个一般是第一个做批量用的
	public static int[] CloneDeimension(this NodeMetadata metadata, int defeim=1)
	{
		var destdim = new int[metadata.Dimensions.Length];
		for (int i = 0; i < destdim.Length; i++)
		{
			if (destdim[i] <= 0)
				destdim[i] = defeim;
		}

		return destdim;
	}
}