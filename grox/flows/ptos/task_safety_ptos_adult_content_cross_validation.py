import logging
from enum import Enum

from grox.core.data_loaders.data_types import Post
from grox.core.schedules.types import TaskContext
from grox.core.tasks.task import Task, TaskResultCategory, TaskWithPost
from grox.flows.ptos.classifier import SafetyPtosAdultContentCrossValidationJudge
from grox.flows.ptos.constants import DELUXE_TIER_TASK_TYPES, SPECIAL_VIDEO_TASK_TYPES
from grox.flows.ptos.state import (
    SafetyPolicy,
    SafetyPolicyCategory,
    SafetyPolicyType,
    SafetyPtosState,
    SafetyPtosViolatedPolicy,
)
from monitor.metrics import Metrics
from grox.flows.ptos.prior_nsfw import post_is_already_flagged_nsfw

logger = logging.getLogger(__name__)

_METRIC_PREFIX = "task.safety_ptos_adult_content_cross_validation"
_CROSS_VALIDATION_REASON = "Grok 4.5 Cross Validation disagreed"
_CROSS_VALIDATION_FAILED_REASON = "Grok 4.5 Cross Validation failed"


class CompareOutcome(str, Enum):
    BOTH_POSITIVE = "both_positive"
    BOTH_NEGATIVE = "both_negative"
    SAFEMODEL_ONLY_POSITIVE = "safemodel_only_positive"
    PTOS_ONLY_POSITIVE = "ptos_only_positive"

    @property
    def is_disagreement(self) -> bool:
        return self in (
            CompareOutcome.SAFEMODEL_ONLY_POSITIVE,
            CompareOutcome.PTOS_ONLY_POSITIVE,
        )


class TaskSafetyPtosAdultContentCrossValidation(TaskWithPost):
    _judge = SafetyPtosAdultContentCrossValidationJudge()

    @classmethod
    async def _exec_with_post(cls, ctx: TaskContext, post: Post) -> None:
        try:
            await cls._run(ctx, post)
        except Exception as e:
            Metrics.counter(f"{_METRIC_PREFIX}.error.count").add(1)
            logger.warning(
                f"Post {post.id}: cross validation failed, keeping original decision: {e}"
            )

    @classmethod
    async def _run(cls, ctx: TaskContext, post: Post) -> None:
        state = ctx.state(SafetyPtosState)
        if not state.safemodel_sex_nudity.scored:
            return

        is_deluxe = ctx.payload.task_type in DELUXE_TIER_TASK_TYPES
        flow = "deluxe" if is_deluxe else "standard"
        safemodel_positive = state.safemodel_sex_nudity.positive
        violations = (
            (state.annotations.violatedPolicies or []) if state.annotations else []
        )
        ptos_positive = any(
            v.category == SafetyPolicyCategory.AdultContent
            and v.safetyPolicy is not None
            and v.safetyPolicy.policyType == SafetyPolicyType.AdultContentSexualHard
            for v in violations
        )

        outcome = cls._compare_outcome(safemodel_positive, ptos_positive)
        Metrics.counter(f"{_METRIC_PREFIX}.compare.count").add(
            1, attributes={"outcome": outcome.value, "flow": flow}
        )
        logger.info(
            f"Post {post.id} ({flow}): safemodel={'positive' if safemodel_positive else 'negative'} "
            f"ptos={'positive' if ptos_positive else 'negative'} outcome={outcome.value}"
        )

        if is_deluxe and outcome.is_disagreement:
            if await post_is_already_flagged_nsfw(ctx, post):
                Metrics.counter(f"{_METRIC_PREFIX}.skipped.count").add(
                    1, attributes={"reason": "prior_nsfw"}
                )
                return
            await cls._cross_validate(ctx, post)

    @staticmethod
    def _compare_outcome(
        safemodel_positive: bool, ptos_positive: bool
    ) -> CompareOutcome:
        if safemodel_positive and ptos_positive:
            return CompareOutcome.BOTH_POSITIVE
        if not safemodel_positive and not ptos_positive:
            return CompareOutcome.BOTH_NEGATIVE
        if safemodel_positive:
            return CompareOutcome.SAFEMODEL_ONLY_POSITIVE
        return CompareOutcome.PTOS_ONLY_POSITIVE

    @classmethod
    async def _cross_validate(cls, ctx: TaskContext, post: Post) -> None:
        metric = f"{_METRIC_PREFIX}.judged.count"
        try:
            judged = await cls._judge.judge(post)
        except Exception as e:
            Metrics.counter(metric).add(1, attributes={"outcome": "error"})
            if ctx.payload.task_type in SPECIAL_VIDEO_TASK_TYPES:
                logger.warning(
                    f"Post {post.id}: grok 4.5 cross validation failed, failing closed to Soft: {e}"
                )
                cls._apply_soft_verdict(ctx, _CROSS_VALIDATION_FAILED_REASON)
            else:
                logger.warning(
                    f"Post {post.id}: grok 4.5 cross validation failed, keeping original decision: {e}"
                )
            return

        is_hard = judged.policyType == SafetyPolicyType.AdultContentSexualHard
        Metrics.counter(metric).add(1, attributes={"outcome": judged.policyType.value})
        logger.info(
            f"Post {post.id}: grok 4.5 cross validation judged {judged.policyType.value}"
        )
        if is_hard:
            return

        cls._apply_soft_verdict(ctx, _CROSS_VALIDATION_REASON)

    @classmethod
    def _apply_soft_verdict(cls, ctx: TaskContext, reason: str) -> None:
        state = ctx.state(SafetyPtosState)
        violations = state.annotations.violatedPolicies or []
        adult_violations = [
            v for v in violations if v.category == SafetyPolicyCategory.AdultContent
        ]
        if not adult_violations:
            adult_violations = [
                SafetyPtosViolatedPolicy(
                    category=SafetyPolicyCategory.AdultContent, reason=reason
                )
            ]
            violations.append(adult_violations[0])
        for violation in adult_violations:
            violation.safetyPolicy = SafetyPolicy(
                policyType=SafetyPolicyType.AdultContentSexualSoft, reason=reason
            )
        state.annotations.violatedPolicies = violations
        state.safemodel_sex_nudity.positive = False

    @classmethod
    async def exec(cls, ctx: TaskContext) -> TaskResultCategory:
        return await Task.exec.__wrapped__(cls, ctx)
