"""Start menu of the viewer app: a page that links to each scenario viewer.

`docs/index.html` lists the scenario pages, one card each, named by the type of
threat. The viewer pages link back to it ("Menu"), and `python -m viz.app` opens it.
It is plain HTML written from `viz/menu_template.html`.

What a card says about its cases (their names and whether each was captured) is
read from the viewer page itself (`viz.viewer.read_viewer`), never written here by
hand, so the menu cannot disagree with the page it links to. A card whose page has
not been generated yet says which example writes it (a page opened from disk cannot
check other files itself).
"""

import html
from dataclasses import dataclass
from pathlib import Path

from viz.viewer import read_viewer

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


@dataclass(frozen=True)
class MenuCase:
    """One case of a page, as the page itself reports it."""

    name: str  # "A · slow threat"
    detail: str  # "8 m/s, turning 1.5 deg/s"
    captured: bool


ENTRIES = (
    MenuEntry(
        kind="Generic target",
        description=(
            "A scripted threat: a point that follows a closed-form path (a gentle "
            "turn or a straight line), with no aircraft behind it."
        ),
        page="iris_scramble_viewer.html",
        command="python -m examples.iris_scramble",
    ),
    MenuEntry(
        kind="Fixed wing",
        description=(
            "A simulated fixed-wing drone (6-DOF, generic small-UAV aerodynamics, "
            "electric propeller) flown by its own autopilot."
        ),
        page="fixed_wing_scramble_viewer.html",
        command="python -m examples.fixed_wing_scramble",
    ),
)


def page_cases(path: Path) -> tuple[MenuCase, ...]:
    """The cases of a viewer page and their results, read from the page."""
    return tuple(
        MenuCase(
            name=scenario.get("name", scenario.get("id", "")),
            detail=scenario.get("detail", ""),
            captured=bool(scenario["captured"]),
        )
        for scenario in read_viewer(path)["scenarios"]
    )


def card(entry: MenuEntry, cases: tuple[MenuCase, ...] | None) -> str:
    """HTML of one card. `cases` is None when the page does not exist yet."""
    e = html.escape
    if cases is None:
        listing = ""
        action = (
            f'      <p class="missing">Not generated yet: run <code>{e(entry.command)}'
            "</code></p>"
        )
    else:
        rows = "\n".join(
            f"        <li><span>{e(case.name)}"
            f"{f' ({e(case.detail)})' if case.detail else ''}</span>"
            f'<b class="badge {"captured" if case.captured else "missed"}">'
            f"{'captured' if case.captured else 'missed'}</b></li>"
            for case in cases
        )
        listing = f'      <ul class="cases">\n{rows}\n      </ul>\n'
        action = (
            f'      <a class="ctl open" href="{e(entry.page)}">Open viewer &rsaquo;</a>'
        )
    return (
        '    <article class="card">\n'
        '      <div class="eyebrow">Threat</div>\n'
        f"      <h2>{e(entry.kind)}</h2>\n"
        f"      <p>{e(entry.description)}</p>\n"
        f"{listing}"
        f"{action}\n"
        "    </article>"
    )


def write_menu(folder: Path, entries: tuple[MenuEntry, ...] = ENTRIES) -> Path:
    """Write the menu page (`index.html`) into `folder`, next to the viewer pages.

    A card whose page exists in `folder` lists the page's cases with their results
    and links to it; otherwise it shows the command that writes the page.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    cards = "\n".join(
        card(e, page_cases(folder / e.page) if (folder / e.page).is_file() else None)
        for e in entries
    )
    path = folder / MENU_FILE
    path.write_text(
        TEMPLATE.read_text(encoding="utf-8").replace(PLACEHOLDER, cards),
        encoding="utf-8",
    )
    return path
