"""Start menu of the viewer app: an index of the scenario pages.

`docs/index.html` lists the scenario pages, one row each, named by the type of
threat and start. The viewer pages link back to it ("Menu"), and `python -m viz.app`
opens it. It is plain HTML written from `viz/menu_template.html`. Which pages it
shows, in what order and with what text comes from the scenario files
(`scenarios/menu.yaml` and each folder's `page.yaml`, read by
`viz.pages.menu_entries`).

What a row says about its cases (their names, how each interceptor -> threat pair
ended and its closest approach) is read from the viewer page itself
(`viz.viewer.read_viewer`), never written here by hand, so the menu cannot disagree
with the page it links to. A row whose page has not been generated yet says which
command writes it (a page opened from disk cannot check other files itself).
"""

import html
from dataclasses import dataclass
from pathlib import Path

from viz.viewer import read_viewer

TEMPLATE = Path(__file__).resolve().parent / "menu_template.html"
PLACEHOLDER = "__MENU_CARDS__"
TALLY = "__MENU_TALLY__"
MENU_FILE = "index.html"

# How an outcome is shown: (colour class, words). Missing, escaping and passing are
# all "the threat got away" (one colour); losing the target to another interceptor
# is not a miss of this one (its own colour).
OUTCOMES = {
    "captured": ("o-captured", "captured"),
    "missed": ("o-missed", "missed"),
    "escaped": ("o-missed", "escaped"),
    "passed": ("o-missed", "passed"),
    "interceptor lost": ("o-missed", "interceptor lost"),
    "target lost": ("o-lost", "target lost"),
    "open": ("o-open", "still open"),
}


@dataclass(frozen=True)
class MenuEntry:
    """One scenario page of the menu."""

    kind: str  # the row's title: the type of threat and start ("Fixed wing · scramble")
    description: str
    page: str  # viewer file name, in the same folder as the menu
    command: str  # the command that writes the page


@dataclass(frozen=True)
class MenuPair:
    """One interceptor -> threat pair of a case, as the page reports it."""

    label: str  # "iris-n → slow" ("" when the case has a single pair)
    outcome: str  # a key of OUTCOMES
    distance: float  # closest approach [m]
    time: float  # when [s]


@dataclass(frozen=True)
class MenuCase:
    """One case of a page, as the page itself reports it."""

    name: str  # "A · slow threat"
    detail: str  # "8 m/s, turning 1.5 deg/s"
    pairs: tuple[MenuPair, ...]

    @property
    def captured(self) -> int:
        """Pairs that ended in a capture."""
        return sum(pair.outcome == "captured" for pair in self.pairs)


def page_cases(path: Path) -> tuple[MenuCase, ...]:
    """The cases of a viewer page and how each pair ended, read from the page."""
    cases = []
    for scenario in read_viewer(path)["scenarios"]:
        if "pairs" in scenario:
            pairs = tuple(
                MenuPair(
                    label=f"{pair['interceptor']} → {pair['target']}",
                    outcome=pair["outcome"],
                    distance=pair["closest"]["distance"],
                    time=pair["closest"]["time"],
                )
                for pair in scenario["pairs"]
            )
        else:  # one pair: the case itself; a run that hit its time limit is open
            outcome = "captured" if scenario["captured"] else "missed"
            if scenario.get("outcome") == "open":
                outcome = "open"
            pairs = (
                MenuPair(
                    label="",
                    outcome=outcome,
                    distance=scenario["closest"]["distance"],
                    time=scenario["closest"]["time"],
                ),
            )
        cases.append(
            MenuCase(
                name=scenario.get("name", scenario.get("id", "")),
                detail=scenario.get("detail", ""),
                pairs=pairs,
            )
        )
    return tuple(cases)


def outcome_cell(pair: MenuPair) -> str:
    style, words = OUTCOMES.get(pair.outcome, ("o-open", pair.outcome))
    return f'<td class="result {style}">{html.escape(words)}</td>'


def cpa_cell(pair: MenuPair) -> str:
    return f'<td class="num">{pair.distance:.2f} m at T+{pair.time:.1f} s</td>'


def case_rows(case: MenuCase) -> str:
    """Table rows of one case: one row, or a title row and one row per pair."""
    e = html.escape
    title = f"<b>{e(case.name)}</b>" + (
        f"<small>{e(case.detail)}</small>" if case.detail else ""
    )
    if len(case.pairs) == 1:
        pair = case.pairs[0]
        return (
            f'        <tr><td class="case">{title}</td>'
            f"{outcome_cell(pair)}{cpa_cell(pair)}</tr>"
        )
    rows = [
        f'        <tr><td class="case" colspan="3">{title}</td></tr>',
        *(
            f'        <tr class="pair"><td class="case"><small>{e(pair.label)}</small>'
            f"</td>{outcome_cell(pair)}{cpa_cell(pair)}</tr>"
            for pair in case.pairs
        ),
    ]
    return "\n".join(rows)


def card(entry: MenuEntry, cases: tuple[MenuCase, ...] | None) -> str:
    """HTML of one page's row. `cases` is None when the page does not exist yet."""
    e = html.escape
    if cases is None:
        about = (
            f'      <p class="missing">Not generated yet: run <code>{e(entry.command)}'
            "</code></p>\n"
        )
        table = ""
    else:
        pairs = [pair for case in cases for pair in case.pairs]
        captured = sum(case.captured for case in cases)
        strip = "".join(
            f'<span class="{OUTCOMES.get(p.outcome, ("o-open",))[0]}"></span>'
            for p in pairs
        )
        about = (
            f'      <div class="strip" aria-hidden="true">{strip}</div>\n'
            f'      <span class="strip-note">{captured} of {len(pairs)} '
            f"{'pair' if len(pairs) == 1 else 'pairs'} captured, "
            f"{len(cases)} {'case' if len(cases) == 1 else 'cases'}</span>\n"
            f'      <a class="ctl open" href="{e(entry.page)}">Open the viewer</a>\n'
        )
        rows = "\n".join(case_rows(case) for case in cases)
        table = (
            "    <table>\n"
            '      <colgroup><col class="c-case"><col class="c-result">'
            '<col class="c-cpa"></colgroup>\n'
            '      <thead><tr><th>Case</th><th>Result</th><th class="num">'
            "Closest approach</th></tr></thead>\n"
            f"      <tbody>\n{rows}\n      </tbody>\n"
            "    </table>\n"
        )
    return (
        '  <article class="page">\n'
        '    <div class="about">\n'
        f"      <h2>{e(entry.kind)}</h2>\n"
        f"      <p>{e(entry.description)}</p>\n"
        f"{about}"
        "    </div>\n"
        f"{table}"
        "  </article>"
    )


def tally(found: list[tuple[MenuCase, ...]], entries: int) -> str:
    """The line under the title: what the menu holds, counted from the pages."""
    if not found:
        return "No scenario page has been generated yet."
    cases = [case for page in found for case in page]
    pairs = sum(len(case.pairs) for case in cases)
    captured = sum(case.captured for case in cases)
    missing = entries - len(found)
    text = (
        f"<b>{len(found)}</b> scenario {'page' if len(found) == 1 else 'pages'}, "
        f"<b>{len(cases)}</b> cases: <b>{captured}</b> of {pairs} interceptor → "
        "threat pairs end in a capture."
    )
    if missing:
        more = "page is" if missing == 1 else "pages are"
        text += f" {missing} more {more} not generated."
    return text


def write_menu(folder: Path, entries: tuple[MenuEntry, ...]) -> Path:
    """Write the menu page (`index.html`) into `folder`, next to the viewer pages.

    A row whose page exists in `folder` lists the page's cases with their results
    and links to it; otherwise it shows the command that writes the page.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    found = {
        e.page: page_cases(folder / e.page)
        for e in entries
        if (folder / e.page).is_file()
    }
    cards = "\n".join(card(e, found.get(e.page)) for e in entries)
    path = folder / MENU_FILE
    page = TEMPLATE.read_text(encoding="utf-8")
    page = page.replace(PLACEHOLDER, cards).replace(
        TALLY, tally(list(found.values()), len(entries))
    )
    path.write_text(page, encoding="utf-8")
    return path
