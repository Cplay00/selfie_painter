"""OpenAI格式API客户端

支持：OpenAI官方、硅基流动、NewAPI、火山方舟等兼容OpenAI格式的服务
"""

import base64
import json
import uuid
import urllib.request
from typing import Dict, Any, Tuple, Optional

from .base_client import BaseApiClient, logger


class OpenAIClient(BaseApiClient):
    """OpenAI格式API客户端"""

    # 格式缓存（base_url → mode），自动降级后用，避免重复试探
    _format_cache: Dict[str, str] = {}

    @classmethod
    def _get_cached_format(cls, base_url: str) -> Optional[str]:
        return cls._format_cache.get(base_url)

    @classmethod
    def _set_cached_format(cls, base_url: str, mode: str) -> None:
        cls._format_cache[base_url] = mode

    def _make_request(
        self,
        prompt: str,
        model_config: Dict[str, Any],
        size: str,
        strength: Optional[float] = None,
        input_image_base64: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """发送OpenAI格式的HTTP请求生成图片"""
        base_url = model_config.get("base_url", "")
        generate_api_key = model_config.get("api_key", "")
        model = model_config.get("model", "")

        raw_format = model_config.get("img2img_format", "auto").strip().lower()
        # 容错：config.toml 可能存了带中文标签的值，模糊提取实际格式名
        img2img_format = raw_format
        for known in ("edits-images", "edits-file"):
            if known in raw_format:
                img2img_format = known
                break
        else:
            # 匹配不到任何已知格式 → 回退自动模式
            if raw_format not in ("", "auto", "edits-images", "edits-file"):
                img2img_format = "auto"
        is_img2img = input_image_base64 is not None

        # ── 构建格式降级链 ──
        format_chain: list[tuple[str, bool, bool, str]] = []
        # 每项：(mode, is_multipart, use_images_array, endpoint)

        if not is_img2img:
            format_chain = [("", False, False, f"{base_url.rstrip('/')}/images/generations")]
        elif img2img_format and img2img_format != "auto":
            # 用户指定格式：只试一种
            user_mode = img2img_format
            endpoint = f"{base_url.rstrip('/')}/images/edits"
            format_chain = [(user_mode, user_mode == "edits-file", user_mode == "edits-images", endpoint)]
        else:
            # 自动模式：用缓存 → legacy → edits-images → edits-file
            cached = self._get_cached_format(base_url)
            if cached:
                ep = f"{base_url.rstrip('/')}/images/edits" if cached in ("edits-images", "edits-file") else f"{base_url.rstrip('/')}/images/generations"
                format_chain = [(cached, cached == "edits-file", cached == "edits-images", ep)]
            else:
                gen_ep = f"{base_url.rstrip('/')}/images/generations"
                edit_ep = f"{base_url.rstrip('/')}/images/edits"
                format_chain = [
                    ("legacy", False, False, gen_ep),
                    ("edits-images", False, True, edit_ep),
                    ("edits-file", True, False, edit_ep),
                ]

            # 若为官方 OpenAI，跳过 legacy 直接从 edits+multipart 开始
            if "api.openai.com" in base_url.lower():
                edit_ep = f"{base_url.rstrip('/')}/images/edits"
                format_chain = [("official", True, False, edit_ep)]

        # ── 获取公共参数（所有格式共用）──
        custom_prompt_add = model_config.get("custom_prompt_add", "")
        negative_prompt_add = model_config.get("negative_prompt_add", "")
        seed = model_config.get("seed", -1)
        seed_enabled = model_config.get("seed_enabled", True)
        guidance_scale = model_config.get("guidance_scale", 7.5)
        guidance_scale_enabled = model_config.get("guidance_scale_enabled", True)
        watermark = model_config.get("watermark", True)
        num_inference_steps = model_config.get("num_inference_steps", 20)
        num_inference_steps_enabled = model_config.get("num_inference_steps_enabled", True)
        prompt_add = prompt + custom_prompt_add
        negative_prompt = negative_prompt_add

        # 代理配置（所有格式共用）
        proxy_config = self._get_proxy_config()

        last_error: Optional[Exception] = None

        for fmt_index, (mode, is_multipart, use_images_array, endpoint) in enumerate(format_chain):
            # 每次重试重建 payload
            payload_dict: Dict[str, Any] = {
                "model": model,
                "prompt": prompt_add,
                "size": size,
                "n": 1,
            }

            if negative_prompt:
                payload_dict["negative_prompt"] = negative_prompt
            if seed is not None and seed != -1 and seed_enabled:
                payload_dict["seed"] = seed

            if "ark.cn-beijing.volces.com" in base_url:
                payload_dict["watermark"] = watermark
            else:
                if guidance_scale_enabled:
                    payload_dict["guidance_scale"] = guidance_scale
                if num_inference_steps_enabled:
                    payload_dict["num_inference_steps"] = num_inference_steps

            # ── 按格式添加图片参数 ──
            if is_img2img:
                if use_images_array:
                    payload_dict["images"] = [{"image_url": self._prepare_image_data_uri(input_image_base64)}]
                elif not is_multipart:
                    payload_dict["image"] = self._prepare_image_data_uri(input_image_base64)
                    if strength is not None:
                        payload_dict["strength"] = strength
                # multipart：image 不在 JSON payload 中

            # ── 平台兼容性处理（仅在自动模式的 legacy 轮次生效）──
            if (not img2img_format or img2img_format == "auto") and mode == "legacy":
                is_siliconflow = "siliconflow" in base_url.lower() or "api.siliconflow.cn" in base_url.lower()
                is_grok = "api.x.ai" in base_url.lower()

                if is_siliconflow:
                    if "size" in payload_dict:
                        payload_dict["image_size"] = payload_dict.pop("size")
                    if "n" in payload_dict:
                        payload_dict["batch_size"] = payload_dict.pop("n")
                    model_lower = model.lower()
                    if "qwen" in model_lower:
                        if "guidance_scale" in payload_dict:
                            payload_dict["cfg"] = payload_dict.pop("guidance_scale")
                        if "image-edit" in model_lower and "image_size" in payload_dict:
                            del payload_dict["image_size"]

                elif is_grok:
                    supported = ["model", "prompt", "n", "response_format"]
                    payload_dict = {k: v for k, v in payload_dict.items() if k in supported}

            # ── 序列化 ──
            if is_multipart:
                text_fields: Dict[str, str] = {}
                for key in ("model", "prompt", "size", "n", "quality", "response_format",
                             "output_format", "output_compression", "background", "moderation", "user"):
                    if key in payload_dict:
                        text_fields[key] = str(payload_dict[key])
                data, boundary = self._build_multipart_body(input_image_base64, text_fields)
                headers: Dict[str, str] = {
                    "Content-Type": f"multipart/form-data; boundary={boundary}",
                    "Accept": "application/json",
                    "Authorization": f"{generate_api_key}",
                }
            else:
                data = json.dumps(payload_dict).encode("utf-8")
                headers = {
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "Authorization": f"{generate_api_key}",
                }

            # ── 详细调试 ──
            verbose_debug = self.action.get_config("components.enable_verbose_debug", False)
            if verbose_debug:
                safe_payload = payload_dict.copy()
                if "image" in safe_payload:
                    safe_payload["image"] = "[BASE64_DATA...]"
                if "images" in safe_payload:
                    safe_payload["images"] = "[IMAGES_DATA...]"
                safe_headers = headers.copy()
                if "Authorization" in safe_headers:
                    auth_val = safe_headers["Authorization"]
                    safe_headers["Authorization"] = "Bearer ***" if auth_val.startswith("Bearer ") else "***"
                logger.info(f"{self.log_prefix} (OpenAI) [{mode}] 请求端点: {endpoint}")
                logger.info(f"{self.log_prefix} (OpenAI) [{mode}] 请求头: {safe_headers}")
                logger.info(
                    f"{self.log_prefix} (OpenAI) [{mode}] 请求体: {json.dumps(safe_payload, ensure_ascii=False, indent=2)}"
                )

            logger.info(
                f"{self.log_prefix} (OpenAI) [{mode}] 发起图片请求: {model}, Prompt: {prompt_add[:30]}... To: {endpoint}"
            )

            # ── 发送请求 ──
            try:
                req = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")

                if proxy_config:
                    proxy_handler = urllib.request.ProxyHandler(
                        {"http": proxy_config["http"], "https": proxy_config["https"]}
                    )
                    opener = urllib.request.build_opener(proxy_handler)
                    timeout = proxy_config.get("timeout", 600)
                else:
                    opener = urllib.request.build_opener()
                    timeout = 600

                with opener.open(req, timeout=timeout) as response:
                    response_status = response.status
                    response_body_bytes = response.read()
                    response_body_str = response_body_bytes.decode("utf-8")
                    cleaned_response = self._clean_response_body(response_body_str)
                    logger.info(
                        f"{self.log_prefix} (OpenAI) [{mode}] 响应: {response_status}. Preview: {cleaned_response[:150]}..."
                    )

                    if verbose_debug:
                        logger.info(f"{self.log_prefix} (OpenAI) [{mode}] 完整响应体: {cleaned_response}")

                    if 200 <= response_status < 300:
                        response_data = json.loads(response_body_str)
                        b64_data = None
                        image_url = None

                        if (
                            isinstance(response_data.get("data"), list)
                            and response_data["data"]
                            and isinstance(response_data["data"][0], dict)
                            and "b64_json" in response_data["data"][0]
                        ):
                            b64_data = response_data["data"][0]["b64_json"]
                            logger.info(f"{self.log_prefix} (OpenAI) [{mode}] 获取到Base64图片数据，长度: {len(b64_data)}")
                            if (not img2img_format or img2img_format == "auto") and is_img2img:
                                self._set_cached_format(base_url, mode)
                            return True, b64_data
                        elif (
                            isinstance(response_data.get("data"), list)
                            and response_data["data"]
                            and isinstance(response_data["data"][0], dict)
                        ):
                            image_url = response_data["data"][0].get("url")
                        elif (
                            isinstance(response_data.get("images"), list)
                            and response_data["images"]
                            and isinstance(response_data["images"][0], dict)
                        ):
                            image_url = response_data["images"][0].get("url")
                        elif response_data.get("url"):
                            image_url = response_data.get("url")

                        if image_url:
                            logger.info(f"{self.log_prefix} (OpenAI) [{mode}] 图片生成成功，URL: {image_url[:70]}...")
                            if (not img2img_format or img2img_format == "auto") and is_img2img:
                                self._set_cached_format(base_url, mode)
                            return True, image_url
                        else:
                            logger.error(
                                f"{self.log_prefix} (OpenAI) [{mode}] API成功但无图片URL: {cleaned_response[:300]}..."
                            )
                            return False, "图片生成API响应成功但未找到图片URL"
                    else:
                        logger.error(
                            f"{self.log_prefix} (OpenAI) [{mode}] API请求失败. 状态: {response.status}. 正文: {cleaned_response[:300]}..."
                        )
                        if fmt_index < len(format_chain) - 1:
                            logger.warning(f"{self.log_prefix} (OpenAI) [{mode}] 失败({response.status})，降级下一格式")
                            last_error = Exception(f"HTTP {response.status}")
                            continue
                        return False, f"图片API请求失败(状态码 {response.status})"

            except urllib.error.HTTPError as e:
                last_error = e
                if fmt_index < len(format_chain) - 1:
                    logger.warning(f"{self.log_prefix} (OpenAI) [{mode}] HTTPError({e.code})，降级下一格式")
                    continue
                logger.error(f"{self.log_prefix} (OpenAI) 所有格式均失败: {e!r}")
                return False, f"所有图生图格式尝试均失败: HTTP {e.code}"
            except Exception as e:
                last_error = e
                if fmt_index < len(format_chain) - 1:
                    logger.warning(f"{self.log_prefix} (OpenAI) [{mode}] 异常({e!r})，降级下一格式")
                    continue
                logger.error(f"{self.log_prefix} (OpenAI) 图片生成时意外错误: {e!r}", exc_info=True)
                return False, f"图片生成HTTP请求时发生意外错误: {str(e)[:100]}"

        return False, f"所有格式尝试均失败: {last_error}"

    def _build_multipart_body(
        self, image_base64: str, fields: Dict[str, str]
    ) -> Tuple[bytes, str]:
        """构建 multipart/form-data 请求体

        Args:
            image_base64: 图片 base64（可能带 data:image/... 前缀）
            fields: 文本字段字典

        Returns:
            (body_bytes, boundary_string)
        """
        clean_b64 = self._get_clean_base64(image_base64)
        image_bytes = base64.b64decode(clean_b64)
        mime_type = self._detect_mime_type(clean_b64)

        ext_map = {
            "image/jpeg": "jpg",
            "image/png": "png",
            "image/webp": "webp",
            "image/gif": "gif",
        }
        ext = ext_map.get(mime_type, "png")
        filename = f"image_{uuid.uuid4().hex[:8]}.{ext}"

        boundary = uuid.uuid4().hex
        body = b""

        # image 文件字段
        body += f"--{boundary}\r\n".encode()
        body += f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'.encode()
        body += f"Content-Type: {mime_type}\r\n\r\n".encode()
        body += image_bytes
        body += b"\r\n"

        # 各文本字段
        for field_name, field_value in fields.items():
            body += f"--{boundary}\r\n".encode()
            body += f'Content-Disposition: form-data; name="{field_name}"\r\n\r\n'.encode()
            body += str(field_value).encode("utf-8")
            body += b"\r\n"

        body += f"--{boundary}--\r\n".encode()
        return body, boundary

    def _clean_response_body(self, response_body: str) -> str:
        """清理响应体中的base64图片数据，避免日志打印完整的base64字符串

        Args:
            response_body: 原始响应体字符串

        Returns:
            清理后的响应体，base64数据被替换为占位符
        """
        try:
            # 如果响应体是JSON，尝试解析并替换b64_json字段
            data = json.loads(response_body)
            if isinstance(data, dict):
                # 检查是否有b64_json字段
                if "data" in data and isinstance(data["data"], list) and len(data["data"]) > 0:
                    for item in data["data"]:
                        if isinstance(item, dict) and "b64_json" in item:
                            item["b64_json"] = "[BASE64_DATA...]"
                # 检查是否有images字段（魔搭格式）
                if "images" in data and isinstance(data["images"], list) and len(data["images"]) > 0:
                    for img in data["images"]:
                        if isinstance(img, dict) and "url" in img:
                            # URL可以保留
                            pass
                # 重新序列化为字符串
                return json.dumps(data, ensure_ascii=False)
        except (json.JSONDecodeError, TypeError):
            # 如果不是JSON，检查是否是纯base64图片数据
            # 常见的base64图片前缀
            base64_prefixes = ["/9j/", "iVBORw", "UklGR", "R0lGOD"]
            if any(response_body.startswith(prefix) for prefix in base64_prefixes):
                return "[BASE64_IMAGE_DATA...]"
            # 如果包含很长的base64字符串（长度>500），截断
            if len(response_body) > 500 and all(
                c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=" for c in response_body[:100]
            ):
                return f"[BASE64_DATA_LEN:{len(response_body)}]"
        # 其他情况返回原样
        return response_body
