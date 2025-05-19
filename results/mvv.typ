#import "template.typ": *

#show: slides.with(
  title: "MVV Graph",

  // Optional Styling (for more / explanation see in the typst universe)
  ratio: 16/9,
  layout: "medium",
  title-color: blue.darken(60%),
  toc: true,
)

= Exercise
== Graph (U-Bahn only)
#align(
  center,
  image("test_construction.png")
)

== Shortest distances constant weight (Stations between)
#align(
  center,
  image("constant_weights.png")
)

== Max. Shortest distances constant weight
#align(
  center,
  image("longest_constant_weight.png")
)

== Shortest distances with second weight (Minutes traveled)
#align(
  center,
  image("seconds_weights.png")
)

== Again has also longest travel time...
#align(
  center,
  image("longest_constant_weight.png")
)

== (Closeness) Centrality
#align(
  center,
  image("closeness_centrality.png")
)

= Discussion

== Transfer times
#align(
  center,
  [
    *Introduce multiple vertices for each station*
    #image("extension.png", height: 90%)
  ]
)

== Possible Extensions

#align(
  horizon,
  [
    1. Include further Transportation devices.
    2. Maybe include shortcut through e-scooter
    3. Include Prices / Zones
    4. Include Statistics how many people traveled when on what routes.
  ]
)
