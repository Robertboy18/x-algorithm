import uuid

from grok_sampler.oai_sampler import OaiSampler
from grox.flows.reply_spam.classifier_reply_ranking import ReplyScorer
from grox.config.config import grox_config
from grox.core.data_loaders.data_types import Post
from grox.core.lm.convo import Conversation, Message, Role
from grox.core.lm.thread import ThreadRenderer
from grox.flows.reply_spam.prompts import reply_scoring_system_simple_prompt
from grox.flows.reply_spam.constants import (
    GEMMA_2,
    GEMMA_REPLY_SPAM,
    GEMMA_REPLY_SPAM_MIN_FOLLOWERS,
)


class SimpleReplyScorer(ReplyScorer):
    def __init__(self):
        self.oai_gemma4 = OaiSampler(grox_config.get_oai_model(GEMMA_2))
        self.oai_gemma4_reply_spam = OaiSampler(
            grox_config.get_oai_model(GEMMA_REPLY_SPAM)
        )

    async def _to_convo(self, post: Post) -> Conversation:
        convo = Conversation(conversation_id=uuid.uuid4().hex)
        system_prompt = reply_scoring_system_simple_prompt(100_000)
        convo.messages.append(Message(role=Role.SYSTEM, content=[system_prompt]))
        convo.messages.append(
            ThreadRenderer.render(
                post, role=Role.HUMAN, include_signals=True, include_follower_count=True
            )
        )
        return convo

    def _sampler(self, post: Post) -> OaiSampler:
        thread_followers = max(
            (p.user.follower_count or 0)
            for p in (post.ancestors[0], post.ancestors[-1])
            if p.user
        )
        return (
            self.oai_gemma4_reply_spam
            if thread_followers > GEMMA_REPLY_SPAM_MIN_FOLLOWERS
            else self.oai_gemma4
        )

    async def _sample(self, convo: Conversation, post: Post) -> str:
        return await self._sampler(post).sample(
            convo.to_openai_messages(), conversation_id=convo.conversation_id
        )
