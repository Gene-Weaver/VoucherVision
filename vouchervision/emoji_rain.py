from typing import Union

import streamlit as st

import inspect
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar, Union, overload

try:
    from streamlit.runtime.metrics_util import gather_metrics as _gather_metrics
except ImportError:

    def _gather_metrics(name, func):  # type: ignore
        return func


F = TypeVar("F", bound=Callable[..., Any])

# Typing overloads here are actually required so that you can correctly (= with correct typing) use the decorator in different ways:
#   1) as a decorator without parameters @extra
#   2) as a decorator with parameters (@extra(foo="bar") but this also refers to empty parameters @extra()
#   3) as a function: extra(my_function)


@overload
def extra(
    func: F,
) -> F:
    ...


@overload
def extra(
    func: None = None,
) -> Callable[[F], F]:
    ...


def extra(
    func: Optional[F] = None,
) -> Union[Callable[[F], F], F]:

    if func:

        filename = inspect.stack()[1].filename
        submodule = Path(filename).parent.name
        extra_name = "streamlit_extras." + submodule
        module = import_module(extra_name)

        if hasattr(module, "__funcs__"):
            module.__funcs__ += [func]  # type: ignore
        else:
            module.__funcs__ = [func]  # type: ignore

        profiling_name = f"{submodule}.{func.__name__}"
        try:
            return _gather_metrics(name=profiling_name, func=func)
        except TypeError:
            # Don't fail on streamlit==1.13.0, which only expects a callable
            pass

    def wrapper(f: F) -> F:
        return f

    return wrapper


@extra
def proportional_rain(
    emoji1: str, 
    count1: int,
    emoji2: str,
    count2: int,
    font_size: int = 64,
    falling_speed: int = 5,
    animation_length: Union[int, str] = "infinite"
):
    """
    Creates a CSS animation where input emojis fall from top to bottom of the screen.
    The proportion of emojis is based on the provided counts.
    """

    if isinstance(animation_length, int):
        animation_length = f"{animation_length}"

    # CSS Code ...
    st.write(
        f"""
    <style>

    body {{
    background: gray;
    }}

    .emoji {{
    color: #777;
    font-size: {font_size}px;
    font-family: Arial;
    // text-shadow: 0 0 5px #000;
    }}

    ///*delete for no hover-effect*/
    //.emoji:hover {{
    //  font-size: 60px;
    //  text-shadow: 5px 5px 5px white;
    //}}

    @-webkit-keyframes emojis-fall {{
    0% {{
        top: -10%;
    }}
    100% {{
        top: 100%;
    }}
    }}
    @-webkit-keyframes emojis-shake {{
    0% {{
        -webkit-transform: translateX(0px);
        transform: translateX(0px);
    }}
    50% {{
        -webkit-transform: translateX(20px);
        transform: translateX(20px);
    }}
    100% {{
        -webkit-transform: translateX(0px);
        transform: translateX(0px);
    }}
    }}
    @keyframes emojis-fall {{
    0% {{
        top: -10%;
    }}
    100% {{
        top: 100%;
    }}
    }}
    @keyframes emojis-shake {{
    0% {{
        transform: translateX(0px);
    }}
    25% {{
        transform: translateX(15px);
    }}
    50% {{
        transform: translateX(-15px);
    }}
    100% {{
        transform: translateX(0px);
    }}
    }}

    .emoji {{
    position: fixed;
    top: -10%;
    z-index: 99999;
    -webkit-user-select: none;
    -moz-user-select: none;
    -ms-user-select: none;
    user-select: none;
    cursor: default;
    -webkit-animation-name: emojis-fall, emojis-shake;
    -webkit-animation-duration: 5s, 3s;
    -webkit-animation-timing-function: linear, ease-in-out;
    -webkit-animation-iteration-count: {animation_length}, {animation_length}; // overall length
    -webkit-animation-play-state: running, running;
    animation-name: emojis-fall, emojis-shake;
    animation-duration: {falling_speed}s, 3s;  // fall speed
    animation-timing-function: linear, ease-in-out;
    animation-iteration-count: {animation_length}, {animation_length}; // overall length
    animation-play-state: running, running;
    }}
    .emoji:nth-of-type(0) {{
    left: 1%;
    -webkit-animation-delay: 0s, 0s;
    animation-delay: 0s, 0s;
    }}
    .emoji:nth-of-type(1) {{
    left: 10%;
    -webkit-animation-delay: 1s, 1s;
    animation-delay: 1s, 1s;
    }}
    .emoji:nth-of-type(2) {{
    left: 20%;
    -webkit-animation-delay: 6s, 0.5s;
    animation-delay: 6s, 0.5s;
    }}
    .emoji:nth-of-type(3) {{
    left: 30%;
    -webkit-animation-delay: 4s, 2s;
    animation-delay: 4s, 2s;
    }}
    .emoji:nth-of-type(4) {{
    left: 40%;
    -webkit-animation-delay: 2s, 2s;
    animation-delay: 2s, 2s;
    }}
    .emoji:nth-of-type(5) {{
    left: 50%;
    -webkit-animation-delay: 8s, 3s;
    animation-delay: 8s, 3s;
    }}
    .emoji:nth-of-type(6) {{
    left: 60%;
    -webkit-animation-delay: 6s, 2s;
    animation-delay: 6s, 2s;
    }}
    .emoji:nth-of-type(7) {{
    left: 70%;
    -webkit-animation-delay: 2.5s, 1s;
    animation-delay: 2.5s, 1s;
    }}
    .emoji:nth-of-type(8) {{
    left: 80%;
    -webkit-animation-delay: 1s, 0s;
    animation-delay: 1s, 0s;
    }}
    .emoji:nth-of-type(9) {{
    left: 90%;
    -webkit-animation-delay: 3s, 1.5s;
    animation-delay: 3s, 1.5s;
    }}

    </style>
    """,
        unsafe_allow_html=True,
    )

    # Create emoji strings based on counts
    emoji_str1 = "".join([f'<div class="emoji">{emoji1}</div>' for _ in range(count1)])
    emoji_str2 = "".join([f'<div class="emoji">{emoji2}</div>' for _ in range(count2)])

    st.write(
        f"""
        <!--get emojis from https://getemoji.com-->
        <div class="emojis">
            {emoji_str1}
            {emoji_str2}
        </div>
        """,
        unsafe_allow_html=True,
    )