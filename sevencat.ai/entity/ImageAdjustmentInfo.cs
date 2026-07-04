namespace sevencat.ai.entity;

public readonly struct ImageAdjustmentInfo(Vector<int> padding, Vector<float> ratio)
{
	public Vector<int> Padding { get; } = padding;

	public Vector<float> Ratio { get; } = ratio;
}