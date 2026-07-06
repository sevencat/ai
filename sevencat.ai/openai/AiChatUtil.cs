using System.ClientModel;
using Microsoft.Extensions.AI;
using OpenAI;
using sevencat.ai.entity;
using sevencat.ai.util;
using sevencat.common;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats;
using SixLabors.ImageSharp.Formats.Jpeg;

namespace sevencat.ai.openai;

public static class AiChatUtil
{
	private static readonly NLog.Logger Log = NLog.LogManager.GetCurrentClassLogger();

	public static OpenAIClient CreateOpenAiClient(string sk, string url)
	{
		var client = new OpenAIClient(
			new ApiKeyCredential(sk),
			new OpenAIClientOptions { Endpoint = new Uri(url) }
		);
		return client;
	}

	public static IChatClient GetChat(this OpenAIClient aiclient, string model)
	{
		return aiclient.GetChatClient(model).AsIChatClient();
	}

	public static async Task<AiTagAnalyzeRet> Analyze(this IChatClient chatclient, Image imgdata,
		ChatPromptBase cp, bool? enablethink = false)
	{
		var fmt = JpegFormat.Instance;
		return await chatclient.Analyze(imgdata, cp, fmt, enablethink);
	}

	public static async Task<AiTagAnalyzeRet> Analyze(this IChatClient chatclient, Image imgdata,
		ChatPromptBase cp, IImageFormat imgfmt, bool? enablethink = false)
	{
		var msgs = new List<ChatMessage>();
		if (cp.SysPrompt.IsNotNullOrWhiteSpace())
			msgs.Add(new ChatMessage(ChatRole.System, cp.SysPrompt));

		var imgurl = await imgdata.ToBase64UrlAsync(imgfmt);
		var message = new ChatMessage(ChatRole.User, new List<AIContent>
		{
			new TextContent(cp.UserPrompt),
			new DataContent(imgurl, imgfmt.DefaultMimeType)
		});
		msgs.Add(message);

		var options = new ChatOptions
		{
			Temperature = 0f,
		};

		if (enablethink != null)
		{
			options.AdditionalProperties = new()
			{
				{ "enable_thinking", enablethink.Value }
			};
		}

		try
		{
			var resp = await chatclient.GetResponseAsync(msgs, options);
			var retmsg = resp.Text;
			Log.Info("ai返回:{0}", retmsg);
			return Analyze(retmsg, cp.Tags);
		}
		catch (Exception ex)
		{
			Log.Error(ex);
			return null;
		}
	}

	public static async Task<AiTagAnalyzeRet> Analyze(this IChatClient chatclient, Image imgdata1, Image imgdata2,
		ChatPromptBase cp, bool? enablethink = false)
	{
		var fmt = JpegFormat.Instance;
		return await chatclient.Analyze(imgdata1, imgdata2, cp, fmt, enablethink);
	}

	public static async Task<AiTagAnalyzeRet> Analyze(this IChatClient chatclient, Image imgdata1, Image imgdata2,
		ChatPromptBase cp, IImageFormat imgfmt, bool? enablethink = false)
	{
		var msgs = new List<ChatMessage>();
		if (cp.SysPrompt.IsNotNullOrWhiteSpace())
			msgs.Add(new ChatMessage(ChatRole.System, cp.SysPrompt));

		var imgurl1 = await imgdata1.ToBase64UrlAsync(imgfmt);
		var imgurl2 = await imgdata2.ToBase64UrlAsync(imgfmt);
		var message = new ChatMessage(ChatRole.User, new List<AIContent>
		{
			new TextContent(cp.UserPrompt),
			new DataContent(imgurl1, imgfmt.DefaultMimeType),
			new DataContent(imgurl2, imgfmt.DefaultMimeType),
		});
		msgs.Add(message);

		var options = new ChatOptions
		{
			Temperature = 0f,
		};
		if (enablethink != null)
		{
			options.AdditionalProperties = new()
			{
				{ "enable_thinking", enablethink.Value }
			};
		}

		try
		{
			var resp = await chatclient.GetResponseAsync(msgs, options);
			var retmsg = resp.Text;
			Log.Info("ai返回:{0}", retmsg);
			return Analyze(retmsg, cp.Tags);
		}
		catch (Exception ex)
		{
			Log.Error(ex);
			return null;
		}
	}


	public static async Task<AiTagAnalyzeRet> Analyze(this IChatClient chatclient, Image[] imgdata,
		ChatPromptBase cp, bool? enablethink = false)
	{
		var fmt = JpegFormat.Instance;
		return await chatclient.Analyze(imgdata, cp, fmt, enablethink);
	}

	public static async Task<AiTagAnalyzeRet> Analyze(this IChatClient chatclient, Image[] imgdata,
		ChatPromptBase cp, IImageFormat imgfmt, bool? enablethink = false)
	{
		var msgs = new List<ChatMessage>();
		if (cp.SysPrompt.IsNotNullOrWhiteSpace())
			msgs.Add(new ChatMessage(ChatRole.System, cp.SysPrompt));

		var aictx = new List<AIContent>() { new TextContent(cp.UserPrompt) };
		foreach (var img in imgdata)
		{
			var imgurl = await img.ToBase64UrlAsync(imgfmt);
			aictx.Add(new DataContent(imgurl, imgfmt.DefaultMimeType));
		}

		var message = new ChatMessage(ChatRole.User, aictx);
		msgs.Add(message);

		var options = new ChatOptions
		{
			Temperature = 0f,
		};
		if (enablethink != null)
		{
			options.AdditionalProperties = new()
			{
				{ "enable_thinking", enablethink.Value }
			};
		}

		try
		{
			var resp = await chatclient.GetResponseAsync(msgs, options);
			var retmsg = resp.Text;
			Log.Info("ai返回:{0}", retmsg);
			return Analyze(retmsg, cp.Tags);
		}
		catch (Exception ex)
		{
			Log.Error(ex);
			return null;
		}
	}

	//请用json返回图片中下列内容：
	//1、是否确定有人抽烟（mwsmoke_exist),以及相关描述(mwsmoke_desc)，
	//2、是否确定有火(fire_exist)以及描述(fire_desc)，
	//3、是否确定有浓烟(smoke_exist)以及描述(smoke_desc),
	//4、是否确定有不带头盔的人（mnhelm_exist)以及描述(mnhelm_desc),
	private const string AIKeyPostfix = "_exist";
	private const string AIDescPostfix = "_desc";

	public static AiTagAnalyzeRet Analyze(string msg, Dictionary<string, string> tags)
	{
		msg = msg.Trim();
		var ret = new AiTagAnalyzeRet();
		msg = msg.Trim().Substring(JSONPREFIX.Length + 1, msg.Length - JSONPREFIX.Length - JSONPOST.Length - 2);
		var jsonmsg = msg.FromJson<Dictionary<string, object>>();
		foreach (var tag in tags)
		{
			var tagk = tag.Key;
			var dictk = tagk + AIKeyPostfix;
			if (!jsonmsg.TryGetValue(dictk, out var dictv))
				continue;
			//解析这个dictv
			var sdictv = dictv?.ToString()?.ToLower();
			if (sdictv is "true" or "1" or "是" or "有")
			{
				//如果检查到有
				var dictdesck = tagk + AIDescPostfix;
				var descv = jsonmsg.GetValueOrDefault(dictdesck).ToString();
				ret.TagItems.Add(new AiTagAnalyzeRet.TagItem()
				{
					Code = tag.Key,
					Name = tag.Value,
					Remark = descv,
				});
			}
		}

		return ret;
	}

	private const string JSONPREFIX = "```json";
	private const string JSONPOST = "```";
}