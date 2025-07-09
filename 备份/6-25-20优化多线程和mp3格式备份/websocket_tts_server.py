import asyncio
import websockets
import json
import os
import torch
import torchaudio
from inference_webui import get_tts_wav
from datetime import datetime, timedelta
import aiohttp
from aiohttp import web
import time
import random
import subprocess
import re
import traceback

# 用户指定的参考路径
REF_AUDIO_PATH = "output/CanKao/CanKao.wav"
REF_TEXT_PATH = "output/CanKao/CanKao_text.txt"
OUTPUT_DIR = "output/tts_results"
EXPIRE_DAYS = 1

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

def debug_log(msg):
    print(f"[DEBUG][{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def read_ref_text():
    if not os.path.exists(REF_TEXT_PATH):
        debug_log(f"参考文本文件未找到: {REF_TEXT_PATH}")
        raise FileNotFoundError(f"参考文本文件未找到: {REF_TEXT_PATH}")
    with open(REF_TEXT_PATH, "r", encoding="utf-8") as f:
        return f.read().strip()

def generate_unique_filename():
    # 时间戳+4位随机数
    ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
    rand = random.randint(1000, 9999)
    return f"{ts}_{rand}"

def detect_language(text):
    zh = re.findall(r'[\u4e00-\u9fff]', text)
    en = re.findall(r'[a-zA-Z]', text)
    if zh and en:
        return "中英混合"
    elif en:
        return "英文"
    else:
        return "中文"

def wav_to_mp3(wav_path, mp3_path):
    if not os.path.exists(wav_path):
        debug_log(f"转码失败，WAV文件不存在: {wav_path}")
        return False
    # ffmpeg在PATH下，或请使用绝对路径
    cmd = f'ffmpeg -y -i "{wav_path}" -codec:a libmp3lame -qscale:a 2 "{mp3_path}"'
    debug_log(f"转码命令: {cmd}")
    res = subprocess.run(cmd, shell=True, capture_output=True)
    if res.returncode != 0:
        debug_log(f"ffmpeg转码出错: {res.stderr.decode('utf-8')}")
        return False
    debug_log(f"转码完成: {mp3_path}")
    return True

def run_tts(text):
    ref_text = read_ref_text()
    lang = detect_language(text)
    ref_lang = detect_language(ref_text)
    debug_log(f"使用参考文本: {ref_text}（{ref_lang}）")
    debug_log(f"开始生成语音: {text}（{lang}）")

    # 生成唯一文件名和路径
    file_stem = generate_unique_filename()
    wav_path = os.path.join(OUTPUT_DIR, f"{file_stem}.wav")
    mp3_path = os.path.join(OUTPUT_DIR, f"{file_stem}.mp3")
    debug_log(f"生成WAV输出路径: {wav_path}")
    debug_log(f"生成MP3输出路径: {mp3_path}")

    try:
        tts_gen = get_tts_wav(
            ref_wav_path=REF_AUDIO_PATH,
            prompt_text=ref_text,
            prompt_language=ref_lang,
            text=text,
            text_language=lang,
            how_to_cut="不切",
            top_k=20,
            top_p=0.6,
            temperature=0.6,
            ref_free=False,
            speed=1.0,
            if_freeze=False,
            inp_refs=None,
            sample_steps=8,
            if_sr=False,
            pause_second=0.3,
        )
        # 这里next有可能抛异常
        sr, audio_int16 = next(tts_gen)
        debug_log(f"get_tts_wav 输出长度: {len(audio_int16)} 采样率: {sr}")

        os.makedirs(os.path.dirname(wav_path), exist_ok=True)
        torchaudio.save(wav_path, torch.tensor(audio_int16).unsqueeze(0).to(torch.int16), sample_rate=sr)
        debug_log(f"torchaudio.save 完毕，文件存在: {os.path.exists(wav_path)} 路径: {wav_path}")

        if not os.path.exists(wav_path):
            raise FileNotFoundError(f"WAV文件保存失败: {wav_path}")

        # 再转MP3
        mp3_ok = wav_to_mp3(wav_path, mp3_path)
        if not mp3_ok:
            raise RuntimeError(f"ffmpeg转码失败: {wav_path} -> {mp3_path}")

        file_size = os.path.getsize(mp3_path) if os.path.exists(mp3_path) else 0
        debug_log(f"推理完成，音频保存至: {mp3_path}, 文件大小: {file_size} 字节")
        # 可选: 删除WAV文件节省空间
        try:
            os.remove(wav_path)
            debug_log(f"已删除临时WAV: {wav_path}")
        except Exception as e:
            debug_log(f"删除WAV失败: {str(e)}")
        return file_stem, file_size

    except Exception as e:
        debug_log(f"TTS推理失败: {str(e)}")
        debug_log(traceback.format_exc())
        if os.path.exists(wav_path):
            os.remove(wav_path)
        if os.path.exists(mp3_path):
            os.remove(mp3_path)
        raise

async def cleanup_expired_files():
    while True:
        now = datetime.now()
        for filename in os.listdir(OUTPUT_DIR):
            file_path = os.path.join(OUTPUT_DIR, filename)
            if os.path.isfile(file_path):
                creation_time = datetime.fromtimestamp(os.path.getctime(file_path))
                if now - creation_time > timedelta(days=EXPIRE_DAYS):
                    try:
                        os.remove(file_path)
                        debug_log(f"删除过期文件: {file_path}")
                    except Exception as e:
                        debug_log(f"删除文件失败: {file_path} {str(e)}")
        await asyncio.sleep(24 * 3600)

async def download_handler(request):
    file_stem = request.match_info['file_id']
    file_path = os.path.join(OUTPUT_DIR, f"{file_stem}.mp3")
    if not os.path.exists(file_path):
        return web.Response(status=404, text="文件不存在或已过期")

    # 检查文件是否过期
    creation_time = datetime.fromtimestamp(os.path.getctime(file_path))
    if datetime.now() - creation_time > timedelta(days=EXPIRE_DAYS):
        try:
            os.remove(file_path)
        except:
            pass
        return web.Response(status=404, text="文件不存在或已过期")
    debug_log(f"下载文件: {file_path}")
    return web.FileResponse(file_path)

async def handle_connection(websocket):
    debug_log("新连接建立")
    await websocket.send(json.dumps({
        "指令": "websock状态",
        "参数": "连接成功"
    }, ensure_ascii=False))

    async for message in websocket:
        debug_log(f"收到消息: {message}")
        try:
            if not message:
                await websocket.send(json.dumps({
                    "指令": "错误",
                    "参数": "收到空消息"
                }, ensure_ascii=False))
                continue

            if message.strip().lower() in ["ping", "PING"]:
                continue

            try:
                msg = json.loads(message)
            except json.JSONDecodeError as e:
                debug_log(f"无效 JSON: {message}")
                await websocket.send(json.dumps({
                    "指令": "错误",
                    "参数": f"JSON 解析失败: {str(e)}"
                }, ensure_ascii=False))
                continue

            cmd = msg.get("指令")
            param = msg.get("参数")

            if cmd == "开始推理":
                if not param:
                    await websocket.send(json.dumps({
                        "指令": "错误",
                        "参数": "参数不能为空"
                    }, ensure_ascii=False))
                    continue
                try:
                    file_stem, file_size = run_tts(param)
                    download_url = f"http://43.159.42.232:8766/download/{file_stem}"
                    await websocket.send(json.dumps({
                        "指令": "推理结果",
                        "参数": {
                            "状态": "成功",
                            "下载链接": download_url,
                            "文件大小": file_size
                        }
                    }, ensure_ascii=False))
                    debug_log(f"已发送下载链接: {download_url}")
                except Exception as e:
                    debug_log(f"推理失败: {str(e)}")
                    await websocket.send(json.dumps({
                        "指令": "错误",
                        "参数": f"推理失败: {str(e)}"
                    }, ensure_ascii=False))
            else:
                await websocket.send(json.dumps({
                    "指令": "错误",
                    "参数": "不支持的指令"
                }, ensure_ascii=False))
        except Exception as e:
            debug_log(f"处理消息时出错: {str(e)}")
            await websocket.send(json.dumps({
                "指令": "错误",
                "参数": str(e)
            }, ensure_ascii=False))

async def main():
    # 启动 HTTP 服务器
    app = web.Application()
    app.add_routes([web.get('/download/{file_id}', download_handler)])
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', 8766)
    await site.start()
    debug_log("HTTP 服务器已启动，监听 http://0.0.0.0:8766")

    # 启动清理任务
    asyncio.create_task(cleanup_expired_files())
    debug_log("文件清理任务已启动")

    # 启动 WebSocket 服务器
    debug_log("WebSocket 服务器正在监听 ws://0.0.0.0:8765")
    async with websockets.serve(handle_connection, "0.0.0.0", 8765):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
