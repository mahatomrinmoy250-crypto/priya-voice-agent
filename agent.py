import asyncio
import logging
import os
import httpx
import json
import pytz
from datetime import datetime
from dotenv import load_dotenv
from typing import Annotated

load_dotenv()

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    RoomInputOptions,
    WorkerOptions,
    cli,
    llm,
)
from livekit.plugins import openai, sarvam, silero
from livekit.plugins import deepgram

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("priya-clinic")

# ── Env vars ──────────────────────────────────────────────────────────────────
GROQ_API_KEY      = os.environ.get("GROQ_API_KEY", "")
DEEPGRAM_API_KEY  = os.environ.get("DEEPGRAM_API_KEY", "")
SARVAM_API_KEY    = os.environ.get("SARVAM_API_KEY", "")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID  = os.environ.get("TELEGRAM_CHAT_ID", "")
CAL_API_KEY       = os.environ.get("CAL_API_KEY", "")
CAL_EVENT_TYPE_ID = os.environ.get("CAL_EVENT_TYPE_ID", "")
CLINIC_NAME       = os.environ.get("CLINIC_NAME", "Aapki Clinic")
DOCTOR_NAME       = os.environ.get("DOCTOR_NAME", "Doctor Sahab")
CLINIC_TIMINGS    = os.environ.get("CLINIC_TIMINGS", "Subah 9 se Sham 6 baje tak")

SYSTEM_PROMPT = f"""Aap Priya hain, {CLINIC_NAME} ki AI receptionist.
Aapka kaam {DOCTOR_NAME} ke liye appointments book karna hai.
Clinic timings: {CLINIC_TIMINGS}.

RULES:
- SIRF Hindi ya Hinglish mein bolna hai. Max 2 sentences.
- Caller ka naam zaroor poochho.
- Appointment ke liye: naam, date (YYYY-MM-DD format), time (HH:MM format) lena hai.
- Appointment book hone par book_appointment function call karo.
- Call khatam hone par end_call function call karo.
- Agar caller kuch aur poochhe toh politely clinic info do."""


# ── Helpers ────────────────────────────────────────────────────────────────────
async def send_telegram(msg: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            await c.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"})
    except Exception as e:
        logger.warning(f"Telegram failed: {e}")


async def book_cal(name: str, date: str, time: str, reason: str = "") -> int:
    url = "https://api.cal.com/v2/bookings"
    headers = {
        "Authorization": f"Bearer {CAL_API_KEY}",
        "cal-api-version": "2024-08-13",
        "Content-Type": "application/json",
    }
    body = {
        "eventTypeId": int(CAL_EVENT_TYPE_ID) if CAL_EVENT_TYPE_ID else 0,
        "start": f"{date}T{time}:00+05:30",
        "attendee": {
            "name": name,
            "email": "patient@clinic.placeholder",
            "timeZone": "Asia/Kolkata",
            "language": "en",
        },
        "bookingFieldsResponses": {"notes": reason or f"Booked via AI. Patient: {name}"},
    }
    try:
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.post(url, headers=headers, json=body)
            logger.info(f"Cal.com response: {r.status_code} {r.text[:200]}")
            return r.status_code
    except Exception as e:
        logger.error(f"Cal booking error: {e}")
        return 500


# ── Tool Context ───────────────────────────────────────────────────────────────
class ClinicTools(llm.ToolContext):
    def __init__(self):
        super().__init__(tools=[])
        self.caller_name = ""
        self.booking_made = False

    @llm.function_tool(description="Book appointment for patient. Call this after getting name, date, time.")
    async def book_appointment(
        self,
        patient_name: Annotated[str, "Full name of the patient"],
        date: Annotated[str, "Date in YYYY-MM-DD format e.g. 2026-03-25"],
        time: Annotated[str, "Time in HH:MM format e.g. 10:30"],
        reason: Annotated[str, "Reason for visit"] = "",
    ) -> str:
        self.caller_name = patient_name
        logger.info(f"[BOOKING] {patient_name} on {date} at {time}")
        status = await book_cal(patient_name, date, time, reason)
        ist = pytz.timezone("Asia/Kolkata")
        now_str = datetime.now(ist).strftime("%d %b %I:%M %p")
        if status in (200, 201):
            self.booking_made = True
            await send_telegram(
                f"<b>Appointment Confirmed!</b>\n"
                f"Patient: {patient_name}\n"
                f"Date: {date} at {time}\n"
                f"Reason: {reason or 'Not specified'}\n"
                f"Clinic: {CLINIC_NAME}\n"
                f"Time: {now_str}"
            )
            return f"Bilkul! {patient_name} ji ka appointment book ho gaya {date} ko {time} baje. Confirmation message bhej diya gaya hai."
        else:
            await send_telegram(
                f"<b>Booking Attempt</b>\nPatient: {patient_name}\nDate: {date} {time}\nStatus: {status}"
            )
            return f"Appointment note kar liya gaya hai {patient_name} ji. Hum confirm karenge."

    @llm.function_tool(description="End the call after goodbye or after booking is confirmed.")
    async def end_call(self) -> str:
        ist = pytz.timezone("Asia/Kolkata")
        now_str = datetime.now(ist).strftime("%d %b %I:%M %p")
        msg = (
            f"<b>Call Ended</b>\nCaller: {self.caller_name or 'Unknown'}\n"
            f"Booking: {'Yes' if self.booking_made else 'No'}\n"
            f"Clinic: {CLINIC_NAME}\nTime: {now_str}"
        )
        await send_telegram(msg)
        return "Dhanyavaad! Aapka din shubh ho. Alvida!"


# ── Agent ──────────────────────────────────────────────────────────────────────
class PriyaAgent(Agent):
    def __init__(self, tools: ClinicTools):
        super().__init__(
            instructions=SYSTEM_PROMPT,
            tools=llm.find_function_tools(tools),
        )
        self._tools = tools

    async def on_enter(self):
        greeting = (
            f"Namaste! Main Priya hoon, {CLINIC_NAME} se. "
            "Aap se milke khushi hui. Main aapki kaise madad kar sakti hoon?"
        )
        await self.session.generate_reply(instructions=f"Say exactly: '{greeting}'")


# ── Entrypoint ─────────────────────────────────────────────────────────────────
async def entrypoint(ctx: JobContext):
    await ctx.connect()
    logger.info(f"[ROOM] Connected: {ctx.room.name}")

    # Notify Telegram on incoming call
    ist = pytz.timezone("Asia/Kolkata")
    now_str = datetime.now(ist).strftime("%d %b %I:%M %p")
    caller_id = "unknown"
    for identity in ctx.room.remote_participants:
        caller_id = identity
        break
    await send_telegram(
        f"<b>Incoming Call!</b>\nFrom: {caller_id}\nTime: {now_str}\nClinic: {CLINIC_NAME}"
    )

    tools = ClinicTools()

    # STT — Deepgram if key present, else Sarvam
    if DEEPGRAM_API_KEY:
        agent_stt = deepgram.STT(
            model="nova-2-general",
            language="hi",
            interim_results=False,
        )
        logger.info("[STT] Using Deepgram")
    else:
        agent_stt = sarvam.STT(
            language="unknown",
            model="saaras:v3",
            mode="translate",
            flush_signal=True,
            sample_rate=16000,
        )
        logger.info("[STT] Using Sarvam")

    # LLM — Groq (fast + free)
    agent_llm = openai.LLM(
        model="llama-3.3-70b-versatile",
        base_url="https://api.groq.com/openai/v1",
        api_key=GROQ_API_KEY,
        max_completion_tokens=120,
    )
    logger.info("[LLM] Using Groq llama-3.3-70b")

    # TTS — Sarvam Bulbul v3
    agent_tts = sarvam.TTS(
        target_language_code="hi-IN",
        model="bulbul:v3",
        speaker="priya",
        speech_sample_rate=24000,
    )
    logger.info("[TTS] Using Sarvam Bulbul v3")

    agent = PriyaAgent(tools=tools)

    session = AgentSession(
        stt=agent_stt,
        llm=agent_llm,
        tts=agent_tts,
        turn_detection="stt",
        min_endpointing_delay=0.3,
        allow_interruptions=True,
    )

    await session.start(
        room=ctx.room,
        agent=agent,
        room_input_options=RoomInputOptions(close_on_disconnect=False),
    )
    logger.info("[AGENT] Priya is live and waiting for caller.")


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="priya-clinic",
        )
    )
