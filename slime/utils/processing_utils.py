import base64
import io
import logging

from transformers import AutoProcessor, AutoTokenizer, PreTrainedTokenizerBase, ProcessorMixin

from .vision_cache import get_vision_cache, image_list_to_hash

logger = logging.getLogger(__name__)


def load_tokenizer(name_or_path: str, **kwargs):
    return AutoTokenizer.from_pretrained(name_or_path, **kwargs)


def load_processor(name_or_path: str, **kwargs):
    try:
        proc = AutoProcessor.from_pretrained(name_or_path, **kwargs)
    except (OSError, ValueError) as e:
        logger.warning(f"Failed to load processor from {name_or_path}: {e}")
        proc = None

    # If HF returned a tokenizer, discard it.
    if isinstance(proc, PreTrainedTokenizerBase) or not isinstance(proc, ProcessorMixin):
        proc = None

    return proc


def prepare_model_inputs(
    prompt,
    tokenizer,
    processor=None,
    metadata=None,
    apply_chat_template_kwargs=None,
    use_vision_cache: bool = False,
):
    """Prepare all inputs for model inference.

    Args:
        prompt: The prompt (may contain image references)
        tokenizer: The tokenizer
        processor: Optional processor for multimodal inputs
        metadata: Optional metadata dict
        apply_chat_template_kwargs: Optional kwargs for chat template
        use_vision_cache: If True, use vision feature cache to avoid redundant preprocessing

    Returns:
        tuple: (input_ids, extra_info)
            - input_ids: Token IDs for the prompt
            - extra_info: Dict with 'images', 'videos', 'multimodal_inputs' (or empty dict)
    """
    tools = metadata.get("tools") if metadata else None
    text_prompt = tokenizer.apply_chat_template(
        prompt,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
        **(apply_chat_template_kwargs or {}),
    )

    if not processor:
        input_ids = tokenizer.encode(text_prompt, add_special_tokens=False)
        return input_ids, {}
    else:
        # temporary solution, will write image utils for slime later
        from qwen_vl_utils import process_vision_info

        images, videos = process_vision_info(prompt)

        # Try to get cached vision features if enabled
        multimodal_inputs = None
        image_hash = None
        
        if use_vision_cache and images:
            # Compute hash for cache lookup
            image_hash = image_list_to_hash(images)
            
            # Try to get from cache
            vision_cache = get_vision_cache()
            cached_features = vision_cache.get(images, image_hash=image_hash)
            
            if cached_features is not None:
                # Cache hit - use cached multimodal_inputs
                multimodal_inputs = cached_features
                logger.debug(f"Using cached vision features for image hash: {image_hash[:16]}...")
        
        # If not cached or cache disabled, process images
        if multimodal_inputs is None:
            # Get input IDs with full prompt (text + multimodal)
            processor_output = processor(text=text_prompt, images=images, videos=videos)
            
            # Extract multimodal tokens (exclude text-related tokens)
            multimodal_inputs = {k: v for k, v in processor_output.items() if k not in ["input_ids", "attention_mask"]}
            
            # Cache the processed features if enabled
            if use_vision_cache and images:
                vision_cache = get_vision_cache()
                vision_cache.put(images, multimodal_inputs, image_hash=image_hash)
                logger.debug(f"Cached vision features for image hash: {image_hash[:16]}...")
        else:
            # For cached features, we still need to get input_ids from processor
            # (text processing is cheap compared to vision processing)
            processor_output = processor(text=text_prompt, images=images, videos=videos)
        
        input_ids = processor_output["input_ids"][0]

        extra_info = {
            "images": images,
            "videos": videos,
            "multimodal_inputs": multimodal_inputs,
        }

        return input_ids, extra_info


def encode_image_for_rollout_engine(image) -> str:
    """Load an image from path, ensure RGB, encode as JPEG base64 string."""
    buffer = io.BytesIO()
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
