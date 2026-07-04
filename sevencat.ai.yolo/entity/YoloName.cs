namespace sevencat.ai.yolo.entity;

public class YoloName
{
	public int Id { get; set; }

	public string Name { get; set; }

	public YoloName()
	{
	}

	public override string ToString()
	{
		return $"{nameof(Id)}: {Id}, {nameof(Name)}: {Name}";
	}

	public YoloName(int id, string name)
	{
		Id = id;
		Name = name;
	}
}