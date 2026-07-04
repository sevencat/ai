namespace sevencat.ai.entity;

public class AiTagAnalyzeRet
{
	public class TagItem
	{
		public string Code { get; set; }
		public string Name { get; set; }
		public string Remark { get; set; }
	}

	public List<TagItem> TagItems { get; set; } = [];
}