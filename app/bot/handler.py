from linebot.v3.messaging import AsyncMessagingApi, ReplyMessageRequest, TextMessage
from linebot.v3.webhooks import MessageEvent, TextMessageContent

from app.agent.core import get_transport_response


async def handle_message(event: MessageEvent, api: AsyncMessagingApi) -> None:
    if not isinstance(event.message, TextMessageContent):
        return

    user_text = event.message.text
    reply = await get_transport_response(user_text)

    await api.reply_message(
        ReplyMessageRequest(
            reply_token=event.reply_token,
            messages=[TextMessage(text=reply)],
        )
    )
