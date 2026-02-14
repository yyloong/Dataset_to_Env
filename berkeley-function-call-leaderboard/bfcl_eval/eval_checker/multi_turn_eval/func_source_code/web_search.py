"""
【改动说明】
此文件将原 web_search.py 中的 SerpAPI 替换为 Serper API
主要改动点：
1. 移除了 serpapi 依赖，改用 requests 直接调用 Serper API
2. 环境变量从 SERPAPI_API_KEY 改为 SERPER_API_KEY
3. API 端点从 SerpAPI 改为 https://google.serper.dev/search
4. 响应格式从 "organic_results" 改为 "organic"
5. 添加了 _region_to_gl 方法用于转换地区代码格式
6. 其他方法（fetch_url_content, _fake_requests_get_error_msg）保持不变，通过继承获得
"""

import os
import random
import time
from typing import Optional

import requests

# 导入原文件的 WebSearchAPI 类
from .web_search_origin import WebSearchAPI as BaseWebSearchAPI


class WebSearchAPI(BaseWebSearchAPI):
    """
    【改动点】继承自 BaseWebSearchAPI，只重写需要修改的方法
    """

    def _region_to_gl(self, region: str) -> Optional[str]:
        """
        【新增方法】将 SerpAPI 格式的地区代码（如 'wt-wt', 'us-en'）转换为 Serper API 的 gl 格式（如 'us'）
        对于 'wt-wt'（无地区）返回 None
        """
        if region == "wt-wt":
            return None
        # 提取国家代码（例如：'us-en' -> 'us'）
        parts = region.split("-")
        if len(parts) >= 2:
            return parts[0]
        return None

    def search_engine_query(
        self,
        keywords: str,
        max_results: Optional[int] = 10,
        region: Optional[str] = "wt-wt",
    ) -> list:
        """
        This function queries the search engine for the provided keywords and region.

        Args:
            keywords (str): The keywords to search for.
            max_results (int, optional): The maximum number of search results to return. Defaults to 10.
            region (str, optional): The region to search in. Defaults to "wt-wt". Possible values include:
                - xa-ar for Arabia
                - xa-en for Arabia (en)
                - ar-es for Argentina
                - au-en for Australia
                - at-de for Austria
                - be-fr for Belgium (fr)
                - be-nl for Belgium (nl)
                - br-pt for Brazil
                - bg-bg for Bulgaria
                - ca-en for Canada
                - ca-fr for Canada (fr)
                - ct-ca for Catalan
                - cl-es for Chile
                - cn-zh for China
                - co-es for Colombia
                - hr-hr for Croatia
                - cz-cs for Czech Republic
                - dk-da for Denmark
                - ee-et for Estonia
                - fi-fi for Finland
                - fr-fr for France
                - de-de for Germany
                - gr-el for Greece
                - hk-tzh for Hong Kong
                - hu-hu for Hungary
                - in-en for India
                - id-id for Indonesia
                - id-en for Indonesia (en)
                - ie-en for Ireland
                - il-he for Israel
                - it-it for Italy
                - jp-jp for Japan
                - kr-kr for Korea
                - lv-lv for Latvia
                - lt-lt for Lithuania
                - xl-es for Latin America
                - my-ms for Malaysia
                - my-en for Malaysia (en)
                - mx-es for Mexico
                - nl-nl for Netherlands
                - nz-en for New Zealand
                - no-no for Norway
                - pe-es for Peru
                - ph-en for Philippines
                - ph-tl for Philippines (tl)
                - pl-pl for Poland
                - pt-pt for Portugal
                - ro-ro for Romania
                - ru-ru for Russia
                - sg-en for Singapore
                - sk-sk for Slovak Republic
                - sl-sl for Slovenia
                - za-en for South Africa
                - es-es for Spain
                - se-sv for Sweden
                - ch-de for Switzerland (de)
                - ch-fr for Switzerland (fr)
                - ch-it for Switzerland (it)
                - tw-tzh for Taiwan
                - th-th for Thailand
                - tr-tr for Turkey
                - ua-uk for Ukraine
                - uk-en for United Kingdom
                - us-en for United States
                - ue-es for United States (es)
                - ve-es for Venezuela
                - vn-vi for Vietnam
                - wt-wt for No region

        Returns:
            list: A list of search result dictionaries, each containing information such as:
            - 'title' (str): The title of the search result.
            - 'href' (str): The URL of the search result.
            - 'body' (str): A brief description or snippet from the search result.

        【重写方法】使用 Serper API 替代 SerpAPI

        改动点：
        1. 环境变量：SERPAPI_API_KEY -> SERPER_API_KEY
        2. API 调用方式：从 serpapi.GoogleSearch 改为 requests.post
        3. API 端点：https://google.serper.dev/search
        4. 请求格式：POST JSON，使用 X-API-KEY header
        5. 响应字段：organic_results -> organic
        6. 地区参数：kl -> gl（需要格式转换）
        """
        backoff = 2  # initial back-off in seconds

        # 【改动点1】环境变量从 SERPAPI_API_KEY 改为 SERPER_API_KEY
        api_key = os.getenv("SERPER_API_KEY")
        if not api_key:
            return {"error": "SERPER_API_KEY environment variable is not set"}

        # 【改动点2】转换地区代码格式（SerpAPI 的 kl 参数 -> Serper 的 gl 参数）
        gl = self._region_to_gl(region)

        # 【改动点3】准备 Serper API 的请求 payload（POST JSON 格式）
        payload = {
            "q": keywords,
            "num": max_results,
        }
        if gl:
            payload["gl"] = gl

        # 【改动点4】使用 X-API-KEY header 进行认证
        headers = {
            "X-API-KEY": api_key,
            "Content-Type": "application/json",
        }

        # Infinite retry loop with exponential backoff（保持原有重试逻辑）
        while True:
            try:
                # 【改动点5】使用 requests.post 调用 Serper API，而不是 serpapi.GoogleSearch
                response = requests.post(
                    "https://google.serper.dev/search",  # 【改动点6】新的 API 端点
                    json=payload,
                    headers=headers,
                    timeout=30,
                )
                response.raise_for_status()
                search_results = response.json()
            except requests.exceptions.HTTPError as e:
                # Handle 429 rate limit errors（保持原有错误处理逻辑）
                if e.response.status_code == 429:
                    wait_time = backoff + random.uniform(0, backoff)
                    error_block = (
                        "*" * 100
                        + f"\n❗️❗️ [WebSearchAPI] Received 429 from Serper API. The number of requests sent using this API key exceeds the hourly throughput limit OR your account has run out of searches. Retrying in {wait_time:.1f} seconds…"
                        + "*" * 100
                    )
                    print(error_block)
                    time.sleep(wait_time)
                    backoff = min(backoff * 2, 120)  # cap the back-off
                    continue
                else:
                    error_block = (
                        "*" * 100
                        + f"\n❗️❗️ [WebSearchAPI] Error from Serper API: {str(e)}. "
                        f"This is not a rate-limit error, so it will not be retried."
                        + "*" * 100
                    )
                    print(error_block)
                    return {"error": str(e)}
            except requests.exceptions.RequestException as e:
                # 对 Serper 的底层网络错误做更细粒度的处理：
                # - 明显的瞬时网络/SSL/超时错误：视为可重试，走指数回退
                # - 其他请求异常：直接返回，不再重试
                msg = str(e)
                transient_substrings = [
                    "Max retries exceeded with url",
                    "Read timed out",
                    "timed out",
                    "Connection aborted",
                    "EOF occurred in violation of protocol",
                    "temporarily unavailable",
                ]
                is_transient = isinstance(
                    e,
                    (
                        requests.exceptions.Timeout,
                        requests.exceptions.ConnectionError,
                        requests.exceptions.SSLError,
                    ),
                ) or any(s in msg for s in transient_substrings)

                if is_transient:
                    wait_time = backoff + random.uniform(0, backoff)
                    error_block = (
                        "*" * 100
                        + "\n❗️❗️ [WebSearchAPI] Transient network/SSL error when calling Serper API "
                        f"(will retry): {msg}\n"
                        f"Retrying in {wait_time:.1f} seconds…" + "*" * 100
                    )
                    print(error_block)
                    time.sleep(wait_time)
                    backoff = min(backoff * 2, 120)
                    continue
                else:
                    error_block = (
                        "*" * 100
                        + f"\n❗️❗️ [WebSearchAPI] Non‑retryable RequestException from Serper API: {msg}"
                        + "*" * 100
                    )
                    print(error_block)
                    return {"error": msg}
            except Exception as e:
                # 其它非 requests 异常，直接返回
                error_block = (
                    "*" * 100
                    + f"\n❗️❗️ [WebSearchAPI] Unexpected error when calling Serper API: {str(e)}"
                    + "*" * 100
                )
                print(error_block)
                return {"error": str(e)}

            # Serper API sometimes returns the error in the payload instead of raising
            if "error" in search_results:
                error_msg = str(search_results["error"])
                if "429" in error_msg or "rate limit" in error_msg.lower():
                    wait_time = backoff + random.uniform(0, backoff)
                    error_block = (
                        "*" * 100
                        + f"\n❗️❗️ [WebSearchAPI] Received 429 from Serper API. The number of requests sent using this API key exceeds the hourly throughput limit OR your account has run out of searches. Retrying in {wait_time:.1f} seconds…"
                        + "*" * 100
                    )
                    print(error_block)
                    time.sleep(wait_time)
                    backoff = min(backoff * 2, 120)
                    continue
                else:
                    return {"error": error_msg}

            break  # Success – no rate-limit error detected

        # 【改动点7】响应字段从 "organic_results" 改为 "organic"
        if "organic" not in search_results:
            return {
                "error": "Failed to retrieve the search results from server. Please try again later."
            }

        search_results = search_results["organic"]

        # Convert the search results to the desired format（结果格式转换逻辑保持不变）
        results = []
        for result in search_results[:max_results]:
            if self.show_snippet:
                results.append(
                    {
                        "title": result["title"],
                        "href": result["link"],
                        "body": result["snippet"],
                    }
                )
            else:
                results.append(
                    {
                        "title": result["title"],
                        "href": result["link"],
                    }
                )

        return results

    # 【说明】以下方法从基类继承，无需重写：
    # - fetch_url_content: 保持不变，直接使用基类实现
    # - _fake_requests_get_error_msg: 保持不变，直接使用基类实现
    # - _load_scenario: 保持不变，直接使用基类实现
    # - __init__: 保持不变，直接使用基类实现
