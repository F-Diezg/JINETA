"""Start menu of the viewer app: a page that links to each scenario viewer.

`docs/index.html` lists the scenario pages, one card each, named by the type of
threat. The viewer pages link back to it ("Menu"), and `python -m viz.app` opens it.
It is plain HTML written from `viz/menu_template.html`; a card whose page has not
been generated yet says which example writes it (a page opened from disk cannot
check other files itself).
"""

import html
from dataclasses import dataclass
from pathlib import Path

TEMPLATE = Path(__file__).resolve().parent / "menu_template.html"
PLACEHOLDER = "__MENU_CARDS__"
MENU_FILE = "index.html"


@dataclass(frozen=True)
class MenuEntry:
    """One scenario page of the menu."""

    kind: str  # type of threat, the card's title ("Generic target", "Fixed wing")
    description: str
    page: str  # viewer file name, in the same folder as the menu
    command: str  # the example that writes the page
    cases: tuple[tuple[str, bool], ...]  # (case name, captured)


ENTRIES = (
    MenuEntry(
        kind="Generic target",
        description=(
            "A scripted threat: a point that follows a closed-form path (a gentle "
            "turn or a straight line), with no aircraft behind it."
        ),
        page="iris_scramble_viewer.html",
        command="python -m examples.iris_scramble",
        cases=(
            ("A · slow, turning, 8 m/s", True),
            ("B · fast, straight, 25 m/s", False),
        ),
    ),
    MenuEntry(
        kind="Fixed wing",
        description=(
            "A simulated fixed-wing drone (6-DOF, generic small-UAV aerodynamics, "
            "electric propeller) flown by its own autopilot."
        ),
        page="fixed_wing_scramble_viewer.html",
        command="python -m examples.fixed_wing_scramble",
        cases=(
            ("A · loitering, orbit at 18 m/s", True),
            ("B · fast pass, 30 m/s", False),
        ),
    ),
)


def card(entry: MenuEntry, available: bool) -> str:
    """HTML of one card."""
    e = html.escape
    cases = "\n".join(
        f"        <li><span>{e(name)}</span>"
        f'<b class="badge {"captured" if captured else "missed"}">'
        f"{'captured' if captured else 'missed'}</b></li>"
        for name, captured in entry.cases
    )
    action = (
        f'      <a class="ctl open" href="{e(entry.page)}">Open viewer &rsaquo;</a>'
        if available
        else f'      <p class="missing">Not generated yet: run <code>{e(entry.command)}'
        "</code></p>"
    )
    return (
        '    <article class="card">\n'
        '      <div class="eyebrow">Threat</div>\n'
        f"      <h2>{e(entry.kind)}</h2>\n"
        f"      <p>{e(entry.description)}</p>\n"
        f'      <ul class="cases">\n{cases}\n      </ul>\n'
        f"{action}\n"
        "    </article>"
    )


def write_menu(folder: Path, entries: tuple[MenuEntry, ...] = ENTRIES) -> Path:
    """Write the menu page (`index.html`) into `folder`, next to the viewer pages.

    A card links to its page if the file exists in `folder`; otherwise it shows the
    command that writes it.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    cards = "\n".join(card(e, (folder / e.page).is_file()) for e in entries)
    path = folder / MENU_FILE
    path.write_text(
        TEMPLATE.read_text(encoding="utf-8").replace(PLACEHOLDER, cards),
        encoding="utf-8",
    )
    return path
