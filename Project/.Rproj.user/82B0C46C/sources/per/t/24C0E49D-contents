# http://applied-r.com/plotting-forecast-data-objects-ggplot/

library(ggplot2)
library(ggthemes)

# theme for forecast data objects
theme.fxdat <- theme_gdocs() +
  theme(plot.title = element_text(size = 15),
        plot.subtitle = element_text(size = 11),
        plot.caption = element_text(size = 9, hjust = 0, vjust = 0, colour = "grey50"),
        axis.title.y = element_text(face = "bold", color = "gray30"),
        axis.title.x = element_text(face = "bold", color = "gray30", vjust = -1),
        panel.background = element_rect(fill = "grey95", colour = "grey75"),
        panel.border = element_rect(colour = "grey75"),
        panel.grid.major.y = element_line(colour = "white"),
        panel.grid.minor.y = element_line(colour = "white", linetype = "dotted"),
        panel.grid.major.x = element_line(colour = "white"),
        panel.grid.minor.x = element_line(colour = "white", linetype = "dotted"),
        strip.background = element_rect(size = 1, fill = "white", colour = "grey75"),
        strip.text.y = element_text(face = "bold"),
        axis.line = element_line(colour = "grey75"))
