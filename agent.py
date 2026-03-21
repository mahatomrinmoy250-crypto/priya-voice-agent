import asyncio, logging, os, httpx
from datetime import datetime
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm
from livekit.agents.voice import VoicePipelineAgent
from livekit.plugins import deepgram, openai, silero, sarvam

logger = logging.getLogger("priya")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY", "")
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY", "")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
CAL_API_KEY = os.environ.get("CAL_API_KEY", "")
CAL_EVENT_TYPE_ID = os.environ.get("CAL_EVENT_TYPE_ID", "")
CLINIC_NAME = os.environ.get("CLINIC_NAME", "Aapki Clinic")
DOCTOR_NAME = os.environ.get("DOCTOR_NAME", "Doctor Sahab")
CLINIC_TIMINGS = os.environ.get("CLINIC_TIMINGS", "Subah 9 se Sham 6 baje tak")

SYSTEM_PROMPT = f"""Aap Priya hain, {CLINIC_NAME} ki AI receptionist.
Aapka kaam {DOCTOR_NAME} ke liye appointments book karna hai.
Clinic timings: {CLINIC_TIMINGS}

Rules:
- SIRF Hindi ya Hinglish mein bolna hai
- Max 2 sentences mein jawab do
- Caller ka naam poochho
- Appointment ke liye naam, date, time lena hai
- Appointment milne par book_appointment call karo
- Goodbye par end_call call karo"""


async def send_telegram(msg):
      if not TELEGRAM_BOT_TOKEN: return
            try:
                      async with httpx.AsyncClient(timeout=10) as c:
                                    await c.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                                                                         json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"})
            except Exception as e:
                      logger.error(f"Telegram: {e}")


async def entrypoint(ctx: JobContext):
      await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    p = await ctx.wait_for_participant()
    logger.info(f"Call from: {p.identity}")
    await send_telegram(f"<b>Incoming Call!</b>\nFrom: {p.identity}\nTime: {datetime.now().strftime('%d %b %I:%M %p')}\nClinic: {CLINIC_NAME}")

    fnc_ctx = llm.FunctionContext()

    @fnc_ctx.ai_callable(description="Book appointment when you have patient name, date YYYY-MM-DD, time HH:MM")
    async def book_appointment(patient_name: str, date: str, time: str, reason: str = "") -> str:
              try:
                            async with httpx.AsyncClient(timeout=15) as c:
                                              r = await c.post("https://api.cal.com/v1/bookings",
                                                                                   headers={"Authorization": f"Bearer {CAL_API_KEY}"},
                                                                                   json={"eventTypeId": int(CAL_EVENT_TYPE_ID), "start": f"{date}T{time}:00",
                                                                                                                   "responses": {"name": patient_name, "email": "patient@clinic.com", "notes": reason},
                                                                                                                   "timeZone": "Asia/Kolkata", "language": "en", "metadata": {}})
                                          await send_telegram(f"<b>Appointment Booked!</b>\nPatient: {patient_name}\nDate: {date} {time}\nReason: {reason or 'General'}")
                            if r.status_code in (200, 201):
                                              return f"Bilkul! {patient_name} ji ka appointment book ho gaya - {date} ko {time} baje."
                                          return "Appointment note kar liya gaya. Hum jald confirm karenge."
except Exception as e:
            logger.error(f"Booking: {e}")
            await send_telegram(f"<b>Appointment Request</b>\n{patient_name} - {date} {time}")
            return "Appointment note kar liya gaya."

    @fnc_ctx.ai_callable(description="End call after booking confirmed or patient says goodbye")
    async def end_call() -> str:
              await send_telegram(f"Call ended - {CLINIC_NAME}")
              return f"Dhanyavaad! {CLINIC_NAME} mein aapka swagat hai. Alvida!"

    agent = VoicePipelineAgent(
              vad=silero.VAD.load(),
              stt=deepgram.STT(model="nova-2-general", language="hi", smart_format=True, interim_results=False),
              llm=openai.LLM(model="llama-3.3-70b-versatile", base_url="https://api.groq.com/openai/v1", api_key=GROQ_API_KEY),
              tts=sarvam.TTS(voice="kavya", language="hi-IN", model="bulbul:v3"),
              fnc_ctx=fnc_ctx,
              chat_ctx=llm.ChatContext().append(role="system", text=SYSTEM_PROMPT),
              min_endpointing_delay=0.3,
              max_endpointing_delay=1.5,
    )
    agent.start(ctx.room, p)
    await asyncio.sleep(1)
    await agent.say(f"Namaste! Main Priya hoon, {CLINIC_NAME} se. Aap se milke khushi hui. Main aapki kaise madad kar sakti hoon?", allow_interruptions=True)


if __name__ == "__main__":
      logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, agent_name="priya-clinic"))
