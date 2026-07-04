namespace sevencat.ai.entity;

public class ChatPromptBase
{
	public string SysPrompt { get; set; }

	public string UserPrompt { get; set; }

	public Dictionary<string, string> Tags { get; set; }
}