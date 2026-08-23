library(arrow)

df <- read_parquet("synthetic_price_data.parquet")

cat("=== Summary statistics: Price ===\n")
print(summary(df$price_eur_mwh))

cat("\n=== Correlation: Generation mix vs Price ===\n")
cor_wind <- cor(df$wind_pct, df$price_eur_mwh)
cor_solar <- cor(df$solar_pct, df$price_eur_mwh)
cor_gas <- cor(df$gas_pct, df$price_eur_mwh)
cat("Wind % vs Price:", round(cor_wind, 3), "\n")
cat("Solar % vs Price:", round(cor_solar, 3), "\n")
cat("Gas % vs Price:", round(cor_gas, 3), "\n")

cat("\n=== Linear regression: Price ~ Wind% + Solar% + Gas% + Hour ===\n")
model <- lm(price_eur_mwh ~ wind_pct + solar_pct + gas_pct + hour, data = df)
print(summary(model))

png("r_price_vs_wind.png", width = 800, height = 500)
plot(df$wind_pct, df$price_eur_mwh,
     main = "Price vs Wind Generation %",
     xlab = "Wind %", ylab = "Price (EUR/MWh)",
     pch = 19, col = rgb(0.2, 0.4, 0.8, 0.4))
abline(lm(price_eur_mwh ~ wind_pct, data = df), col = "red", lwd = 2)
dev.off()

cat("\nPlot saved as r_price_vs_wind.png\n")