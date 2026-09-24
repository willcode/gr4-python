"""Changes a setting of a running graph and prints the events the graph reports.

    python3 live_settings.py

A tone runs into a null sink until the program stops it. The program subscribes to the tone's settings, stages a new
frequency three times, and prints each notification the tone sends when it applies one.
"""

import gr4

FREQUENCIES = (880.0, 1320.0, 1760.0)


def describe(event: gr4.Event) -> str:
    """Returns one line about an event: the frequency a settings notification carries, or the whole event."""
    if event.endpoint == "Settings" and not event.is_error:
        return f"{event.endpoint} {event.command.name}: tone_frequency = {event.data['tone_frequency']}"
    return str(event)


def main() -> None:
    """Runs the graph, changes the frequency while it runs, and stops it."""
    graph = gr4.Graph()
    tone = graph.add("gr::blocks::basic::SignalGenerator<float32>", "tone", sample_rate=100_000, tone_frequency=440)
    graph.connect(tone, graph.add("gr::blocks::testing::NullSink<float32>", "sink"))
    scheduler = gr4.Scheduler(graph, "gr::scheduler::Simple<multiThreaded>")
    with scheduler.start(), tone.subscribe("Settings"):
        print("running, tone_frequency =", tone.request("Settings")["tone_frequency"])
        for frequency in FREQUENCIES:
            tone.settings["tone_frequency"] = frequency
            print("staged", frequency)
            for event in scheduler.events(timeout=0.2):
                print(" ", describe(event))
    print("stopped:", scheduler.state.name, "errors:", len(scheduler.errors))
    print("in force:", tone.settings["tone_frequency"])


if __name__ == "__main__":
    main()
