import asyncio
import websockets
import json
import os
import torch
import torchaudio
from inference_webui import get_tts_wav
from datetime import datetime, timedelta
import re
import uuid
import aiohttp
from aiohttp import web

# 固定模型路径（启动时由 inference_webui.py 读取）
REF_AUDIO_PATH = "output/CanKao/CanKao.wav"
REF_TEXT_PATH = "output/CanKao/CanKao_text.txt"
OUTPUT_DIR = "output/tts_results"
EXPIRE_DAYS = 1

os.makedirs(OUTPUT_DIR, exist_ok=True)

def read_ref_text():
    if not os.path.exists(REF_TEXT_PATH):
        raise FileNotFoundError(f"参考文本文件未找到: {REF_TEXT_PATH}")
    with open(REF_TEXT_PATH, "r", encoding="utf-8") as f:
        return f.read().strip()

def detect_language(text):
    zh = re.findall(r'[\u4e00-\u9fff]', text)
    en = re.findall(r'[a-zA-Z]', text)
    if zh and en:
        return "中英混合"
    elif en:
        return "英文"
    else:
        return "中文"

def generate_unique_filename():
    # 时间戳+随机串
    now = datetime.now().strftime('%Y%m%d_%H%M%S')
    rand = str(uuid.uuid4())[:8]
    file_id = f"{now}_{rand}"
    output_path = os.path.join(OUTPUT_DIR, f"{file_id}.wav")
    return file_id, output_path

def run_tts(text):
    ref_text = read_ref_text()
    lang = detect_language(text)
    ref_lang = detect_language(ref_text)
    print(f"[TTS] 使用参考文本: {ref_text}（{ref_lang}）")
    print(f"[TTS] 开始生成语音: {text}（{lang}）")

    file_id, output_path = generate_unique_filename()

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
        sr, audio_int16 = next(tts_gen)
        torchaudio.save(output_path, torch.tensor(audio_int16).unsqueeze(0).to(torch.int16), sample_rate=sr)
        # wav转mp3
        mp3_path = output_path.replace('.wav', '.mp3')
        ffmpeg_cmd = f'ffmpeg -y -i "{output_path}" -codec:a libmp3lame -qscale:a 2 "{mp3_path}"'
        print(f"[DEBUG] ffmpeg转码: {ffmpeg_cmd}")
        os.system(ffmpeg_cmd)
        # 检查mp3存在
        if not os.path.exists(mp3_path):
            raise RuntimeError(f"ffmpeg转码失败: {output_path} -> {mp3_path}")
        os.remove(output_path)
        file_size = os.path.getsize(mp3_path)
        print(f"[TTS] 推理完成，音频保存至: {mp3_path}, 文件大小: {file_size} 字节")
        return os.path.splitext(os.path.basename(mp3_path))[0], file_size, text
    except Exception as e:
        print(f"[TTS] 推理失败: {str(e)}")
        if os.path.exists(output_path):
            os.remove(output_path)
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
                        print(f"[Cleanup] 删除过期文件: {file_path}")
                    except Exception as e:
                        print(f"[Cleanup] 删除文件 {file_path} 失败: {str(e)}")
        await asyncio.sleep(24 * 3600)

async def download_handler(request):
    file_id = request.match_info['file_id']
    file_path = os.path.join(OUTPUT_DIR, f"{file_id}.mp3")
    print(f"[DEBUG] 下载文件: {file_path}")
    if not os.path.exists(file_path):
        return web.Response(status=404, text="文件不存在或已过期")
    creation_time = datetime.fromtimestamp(os.path.getctime(file_path))
    if datetime.now() - creation_time > timedelta(days=EXPIRE_DAYS):
        try:
            os.remove(file_path)
        except:
            pass
        return web.Response(status=404, text="文件不存在或已过期")
    return web.FileResponse(file_path)

async def handle_connection(websocket):
    print("[Server] 新连接建立")
    await websocket.send(json.dumps({
        "指令": "websock状态",
        "参数": "连接成功"
    }, ensure_ascii=False))

    async for message in websocket:
        print(f"[Server] 收到消息: {message}")
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
                print(f"[Server] 无效 JSON: {message}")
                await websocket.send(json.dumps({
                    "指令": "错误",
                    "参数": f"JSON 解析失败: {str(e)}"
                }, ensure_ascii=False))
                continue

            cmd = msg.get("指令")
            param = msg.get("参数")

            if cmd == "开始推理":
                # 只支持传字符串文本（忽略“模型名”参数）
                if not param or not isinstance(param, str):
                    await websocket.send(json.dumps({
                        "指令": "错误",
                        "参数": "参数不能为空，且必须为字符串文本"
                    }, ensure_ascii=False))
                    continue
                try:
                    mp3_id, file_size, text = run_tts(param)
                    download_url = f"http://43.159.42.232:8766/download/{mp3_id}"
                    await websocket.send(json.dumps({
                        "指令": "推理结果",
                        "参数": {
                            "状态": "成功",
                            "推理文本": text,
                            "推理模型": "YMX2",
                            "下载链接": download_url,
                            "文件大小": file_size
                        }
                    }, ensure_ascii=False))
                    print(f"[Server] 已发送下载链接: {download_url}")
                except Exception as e:
                    print(f"[Server] 推理失败: {str(e)}")
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
            print(f"[Server] 处理消息时出错: {str(e)}")
            await websocket.send(json.dumps({
                "指令": "错误",
                "参数": str(e)
            }, ensure_ascii=False))

async def main():
    # HTTP服务
    app = web.Application()
    app.add_routes([
        web.get('/download/{file_id}', download_handler),
    ])
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', 8766)
    await site.start()
    print("[Server] HTTP 服务器已启动，监听 http://0.0.0.0:8766")

    # 文件清理
    asyncio.create_task(cleanup_expired_files())
    print("[Server] 文件清理任务已启动")

    # WebSocket
    print("[Server] WebSocket 服务器正在监听 ws://0.0.0.0:8765")
    async with websockets.serve(handle_connection, "0.0.0.0", 8765):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
