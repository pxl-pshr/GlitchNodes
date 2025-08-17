from .ascii import ASCII
from .Corruptor import Corruptor
from .DataBend import DataBend
from .DitherMe import DitherMe
from .FrequencyModulation import FrequencyModulation
from .GlitchIT import GlitchIT
from .interference import interference
from .LineScreen import LineScreen
from .LuminousFlow import LuminousFlow
from .OrderedDithering import OrderedDithering
from .Pixel8Bit import Pixel8Bit
from .PixelFloat import PixelFloat
from .PixelRedistribution import PixelRedistribution
from .Rekked import Rekked
from .Scanz import Scanz
from .TvGlitch import TvGlitch
from .VaporWave import VaporWave
from .VHSonAcid import VHSonAcid
from .VideoModulation import VideoModulation

NODE_CLASS_MAPPINGS = {
    "ASCII": ASCII,
    "Corruptor": Corruptor,
    "DataBend": DataBend,
    "DitherMe": DitherMe,
    "FrequencyModulation": FrequencyModulation,
    "GlitchIT": GlitchIT,
    "interference": interference,
    "LineScreen": LineScreen,
    "LuminousFlow": LuminousFlow,
    "OrderedDithering": OrderedDithering,
    "Pixel8Bit": Pixel8Bit,
    "PixelFloat": PixelFloat,
    "PixelRedistribution": PixelRedistribution,
    "Rekked": Rekked,
    "Scanz": Scanz,
    "TvGlitch": TvGlitch,
    "VaporWave": VaporWave,
    "VHSonAcid": VHSonAcid,
    "VideoModulation": VideoModulation,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ASCII": "ASCII | GlitchNodes",
    "Corruptor": "Corruptor | GlitchNodes",
    "DataBend": "DataBend | GlitchNodes",
    "DitherMe": "DitherMe | GlitchNodes",
    "FrequencyModulation": "Frequency Modulation WIP | GlitchNodes",
    "GlitchIT": "GlitchIT | GlitchNodes",
    "interference": "interference | GlitchNodes",
    "LineScreen": "LineScreen | GlitchNodes",
    "LuminousFlow": "LuminousFlow | GlitchNodes",
    "OrderedDithering": "Ordered Dithering | GlitchNodes",
    "Pixel8Bit": "8-Bit / Pixelate | GlitchNodes",
    "PixelFloat": "PixelFloat | GlitchNodes",
    "PixelRedistribution": "PixelRedistribution WIP | GlitchNodes",
    "Rekked": "Rekked WIP | GlitchNodes",
    "Scanz": "Scanz | GlitchNodes",
    "TvGlitch": "TV Glitch | GlitchNodes",
    "VaporWave": "VaporWave | GlitchNodes",
    "VHSonAcid": "VHSonAcid | GlitchNodes",
    "VideoModulation": "VideoModulation | GlitchNodes",
}

def print_red(text):
    # ANSI escape code for red text
    RED = '\033[31m'
    # ANSI escape code to reset color
    RESET = '\033[0m'
    print(f"{RED}{text}{RESET}")

ascii_art = """

                      :::!~!!!!!:.
                  .xUHWH!! !!?M88WHX:.
                .X*#M@$!!  !X!M$$$$$$WWx:.
               :!!!!!!?H! :!$!$$$$$$$$$$8X:
              !!~  ~:~!! :~!$!#$$$$$$$$$$8X:
             :!~::!H!<   ~.U$X!?R$$$$$$$$MM!
             ~!~!!!!~~ .:XW$$$U!!?$$$$$$RMM!
               !:~~~ .:!M"T#$$$$WX??#MRRMMM!
               ~?WuxiW*`   `"#$$$$8!!!!??!!!
             :X- M$$$$       `"T#$T~!8$WUXU~
            :%`  ~#$$$m:        ~!~ ?$$$$$$
          :!`.-   ~T$$$$8xx.  .xWW- ~""##*"
.....   -~~:<` !    ~?T#$$@@W@*?$$      /`
W$@@M!!! .!~~ !!     .:XUW$W!~ `"~:    :
#"~~`.:x%`!!  !H:   !WM$$$$Ti.: .!WUn+!`
:::~:!!`:X~ .: ?H.!u "$$$B$$$!W:U!T$$M~
.~~   :X@!.-~   ?@WTWo("*$$$W$TH$! `
Wi.~!X$?!-~    : ?$$$B$Wu("**$RM!
$R@i.~~ !     :   ~$$$$$B$$en:``
?MXT@Wx.~    :     ~"##*$$$$M~                                                                   
▒█▀▀█ ▀▄▒▄▀ ▒█░░░ ▒█▀▀█ ▒█▀▀▀█ ▒█░▒█ ▒█▀▀█ 
▒█▄▄█ ░▒█░░ ▒█░░░ ▒█▄▄█ ░▀▀▀▄▄ ▒█▀▀█ ▒█▄▄▀ 
▒█░░░ ▄▀▒▀▄ ▒█▄▄█ ▒█░░░ ▒█▄▄▄█ ▒█░▒█ ▒█░▒█
~~~~~~~~~ G L I T C H N O D E S ~~~~~~~~~
"""
print_red(ascii_art)