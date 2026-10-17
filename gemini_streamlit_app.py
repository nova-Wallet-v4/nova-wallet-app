from __future__ import annotations

import json
import os
from io import BytesIO

import streamlit as st
from google import genai
from google.genai import types
from PIL import Image, ImageDraw

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

st.set_page_config(
    page_title="Gemini API Playground",
    page_icon="✦",
    layout="wide",
)


def get_store_hours(location: str) -> dict[str, str]:
    """Mock application function. Replace this with a real database/API call."""
    hours = {
        "downtown": "Mon-Sat 08:00-18:00; Sun 09:00-15:00",
        "airport": "Daily 06:00-22:00",
    }
    return {
        "location": location,
        "hours": hours.get(location.lower(), "Location not found"),
    }


def make_sample_image() -> bytes:
    image = Image.new("RGB", (640, 360), "#eef6ff")
    draw = ImageDraw.Draw(image)

    draw.rounded_rectangle(
        (30, 30, 610, 330),
        radius=24,
        fill="#ffffff",
        outline="#2b6cb0",
        width=5,
    )
    draw.ellipse(
        (80, 90, 230, 240),
        fill="#f6ad55",
        outline="#9c4221",
        width=4,
    )
    draw.rectangle(
        (300, 105, 530, 220),
        fill="#68d391",
        outline="#276749",
        width=4,
    )

    draw.text((90, 260), "ORANGE", fill="#7b341e")
    draw.text((350, 245), "GREEN BOX", fill="#22543d")

    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


@st.cache_resource(show_spinner=False)
def create_client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)


def show_error(error: Exception) -> None:
    st.error(f"Gemini request failed: {error}")
    st.caption(
        "Check GEMINI_API_KEY, model availability, quota, and network access."
    )


def chat_contents(messages: list[dict]) -> list[types.Content]:
    """Convert display messages into Gemini's multi-turn Content format."""
    return [
        types.Content(
            role="model" if message["role"] == "assistant" else "user",
            parts=[types.Part(text=message["content"])]
            + [
                types.Part.from_bytes(
                    data=attachment["data"],
                    mime_type=attachment["mime_type"],
                )
                for attachment in message.get("attachments", [])
            ],
        )
        for message in messages
    ]


st.title("Gemini API Playground")
st.write(
    "An interactive version of the command-line demo: "
    "chat, text, JSON, vision, and tools."
)


with st.sidebar:
    st.header("Settings")

    env_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    api_key = env_key or st.text_input(
        "Gemini API key",
        type="password",
    )

    model = st.text_input(
        "Model",
        value=DEFAULT_MODEL,
    )

    if env_key:
        st.caption(
            "Using the GEMINI_API_KEY/GOOGLE_API_KEY environment variable."
        )
    else:
        st.caption(
            "For deployment, prefer an environment variable "
            "or Streamlit secret."
        )


if not api_key:
    st.info("Add a Gemini API key in the sidebar to enable the demos.")
    st.stop()


client = create_client(api_key)


if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []


chat_tab, text_tab, json_tab, image_tab, tools_tab = st.tabs(
    [
        "Chat",
        "Text generation",
        "Structured JSON",
        "Image understanding",
        "Function calling",
    ]
)


with chat_tab:
    st.subheader("Multi-turn conversation")
    st.caption(
        "Messages and small attachments are kept in this browser session "
        "and sent as context on each turn."
    )

    clear_col, count_col = st.columns([1, 4])

    with clear_col:
        if st.button("Clear chat", key="clear_chat_button"):
            st.session_state.chat_messages = []
            st.rerun()

    with count_col:
        st.caption(
            f"{len(st.session_state.chat_messages)} messages in memory"
        )

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            for attachment in message.get("attachments", []):
                if attachment["mime_type"].startswith("image/"):
                    st.image(
                        attachment["data"],
                        caption=attachment["name"],
                        width="stretch",
                    )
                else:
                    st.caption(
                        f"Attached file: {attachment['name']}"
                    )

    chat_turn = st.chat_input(
        "Message Gemini or attach files...",
        accept_file="multiple",
        file_type=[
            "png",
            "jpg",
            "jpeg",
            "webp",
            "pdf",
            "txt",
            "md",
            "csv",
            "json",
        ],
        key="multimodal_chat_input",
    )

    if chat_turn:
        prompt_text = (
            chat_turn.text.strip()
            or "Please analyze the attached file(s)."
        )

        attachments = [
            {
                "name": uploaded_file.name,
                "mime_type": (
                    uploaded_file.type
                    or "application/octet-stream"
                ),
                "data": uploaded_file.getvalue(),
            }
            for uploaded_file in chat_turn.files
        ]

        total_attachment_bytes = sum(
            len(attachment["data"])
            for attachment in attachments
        )

        if total_attachment_bytes > 19 * 1024 * 1024:
            st.error(
                "Please keep chat attachments below 19 MB total "
                "for inline multimodal input."
            )
            st.stop()

        user_message = {
            "role": "user",
            "content": prompt_text,
            "attachments": attachments,
        }

        st.session_state.chat_messages.append(user_message)

        with st.chat_message("user"):
            st.markdown(prompt_text)

            for attachment in attachments:
                if attachment["mime_type"].startswith("image/"):
                    st.image(
                        attachment["data"],
                        caption=attachment["name"],
                        width="stretch",
                    )
                else:
                    st.caption(
                        f"Attached file: {attachment['name']}"
                    )

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = client.models.generate_content(
                        model=model,
                        contents=chat_contents(
                            st.session_state.chat_messages
                        ),
                    )

                    answer = (
                        response.text
                        or "Gemini returned an empty response."
                    )

                    st.markdown(answer)

                    st.session_state.chat_messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                        }
                    )

                except Exception as error:
                    # Keep the failed user message out of history
                    # so it can be retried.
                    st.session_state.chat_messages.pop()
                    show_error(error)


with text_tab:
    st.subheader("Text generation")

    prompt = st.text_area(
        "Prompt",
        value=(
            "In two concise sentences, explain why APIs are useful "
            "to a small business."
        ),
        height=120,
    )

    if st.button(
        "Generate text",
        type="primary",
        key="text_button",
    ):
        with st.spinner("Generating..."):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                )
                st.write(response.text)
            except Exception as error:
                show_error(error)


with json_tab:
    st.subheader("Schema-constrained JSON")

    note = st.text_area(
        "Business note to extract",
        value=(
            "Northstar Bakery is a neighborhood bakery. "
            "It wants to reduce food waste, improve online ordering, "
            "and understand which pastries sell best."
        ),
        height=140,
    )

    if st.button(
        "Extract JSON",
        type="primary",
        key="json_button",
    ):
        schema = {
            "type": "OBJECT",
            "properties": {
                "business_name": {
                    "type": "STRING",
                },
                "industry": {
                    "type": "STRING",
                },
                "priorities": {
                    "type": "ARRAY",
                    "items": {
                        "type": "STRING",
                    },
                },
                "recommended_next_step": {
                    "type": "STRING",
                },
            },
            "required": [
                "business_name",
                "industry",
                "priorities",
                "recommended_next_step",
            ],
        }

        with st.spinner("Extracting structured data..."):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=(
                        "Extract a structured profile from this note: "
                        f"{note}"
                    ),
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=schema,
                    ),
                )

                st.json(json.loads(response.text))

            except Exception as error:
                show_error(error)


with image_tab:
    st.subheader("Multimodal image understanding")

    uploaded = st.file_uploader(
        "Upload an image",
        type=[
            "png",
            "jpg",
            "jpeg",
            "webp",
        ],
    )

    image_bytes = (
        uploaded.getvalue()
        if uploaded
        else make_sample_image()
    )

    st.image(
        image_bytes,
        caption="Image sent to Gemini",
        width="stretch",
    )

    image_prompt = st.text_input(
        "Image prompt",
        value=(
            "Describe the objects, their colors, and the words "
            "visible in this image."
        ),
    )

    if st.button(
        "Analyze image",
        type="primary",
        key="image_button",
    ):
        mime_type = (
            uploaded.type
            if uploaded
            else "image/png"
        )

        with st.spinner("Analyzing image..."):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=[
                        image_prompt,
                        types.Part.from_bytes(
                            data=image_bytes,
                            mime_type=mime_type,
                        ),
                    ],
                )

                st.write(response.text)

            except Exception as error:
                show_error(error)


with tools_tab:
    st.subheader("Function calling")

    location = st.selectbox(
        "Store location",
        [
            "downtown",
            "airport",
        ],
    )

    tool_prompt = st.text_input(
        "Request",
        value="What are the opening hours for our downtown store?",
    )

    if st.button(
        "Ask Gemini to use the tool",
        type="primary",
        key="tool_button",
    ):
        declaration = {
            "name": "get_store_hours",
            "description": (
                "Look up the opening hours for one bakery location."
            ),
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "location": {
                        "type": "STRING",
                        "description": "The store location.",
                    }
                },
                "required": [
                    "location",
                ],
            },
        }

        tool = types.Tool(
            function_declarations=[
                declaration,
            ]
        )

        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part(text=tool_prompt),
                ],
            )
        ]

        with st.spinner("Choosing and executing the tool..."):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        tools=[
                            tool,
                        ]
                    ),
                )

                function_call = next(
                    (
                        part.function_call
                        for part in response.candidates[0]
                        .content.parts
                        if part.function_call
                    ),
                    None,
                )

                if function_call is None:
                    st.write(response.text)

                else:
                    args = dict(function_call.args or {})
                    args["location"] = location

                    result = get_store_hours(**args)

                    st.code(
                        json.dumps(
                            {
                                "requested": args,
                                "tool_result": result,
                            },
                            indent=2,
                        ),
                        language="json",
                    )

                    contents.extend(
                        [
                            response.candidates[0].content,
                            types.Content(
                                role="user",
                                parts=[
                                    types.Part(
                                        function_response=(
                                            types.FunctionResponse(
                                                name=function_call.name,
                                                id=function_call.id,
                                                response={
                                                    "result": result,
                                                },
                                            )
                                        )
                                    )
                                ],
                            ),
                        ]
                    )

                    final_response = client.models.generate_content(
                        model=model,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            tools=[
                                tool,
                            ]
                        ),
                    )

                    st.write(final_response.text)

            except Exception as error:
                show_error(error)


st.divider()

st.caption(
    "Keep API keys server-side. In production, authenticate tool calls "
    "and validate model output before taking actions."
)
