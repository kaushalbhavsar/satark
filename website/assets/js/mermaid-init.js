(() => {
  const diagrams = document.querySelectorAll(".mermaid");
  if (!diagrams.length || typeof mermaid === "undefined") return;

  mermaid.initialize({
    startOnLoad: false,
    securityLevel: "strict",
    theme: "dark",
    fontFamily: '"IBM Plex Sans", "Segoe UI", sans-serif',
    flowchart: {
      curve: "basis",
      padding: 16,
      nodeSpacing: 36,
      rankSpacing: 42,
      htmlLabels: true,
    },
    themeVariables: {
      darkMode: true,
      background: "transparent",
      primaryColor: "#161618",
      primaryTextColor: "#f4f4f5",
      primaryBorderColor: "#ed1c24",
      secondaryColor: "#1d1d21",
      secondaryTextColor: "#f4f4f5",
      secondaryBorderColor: "#ff6b72",
      tertiaryColor: "#121214",
      tertiaryTextColor: "#f4f4f5",
      tertiaryBorderColor: "#e0b35a",
      lineColor: "#ff4d54",
      textColor: "#f4f4f5",
      mainBkg: "#161618",
      nodeBorder: "#ed1c24",
      clusterBkg: "rgba(22, 22, 24, 0.72)",
      clusterBorder: "#ed1c24",
      titleColor: "#ff6b72",
      edgeLabelBackground: "#0c0c0e",
      fontSize: "15px",
    },
  });

  mermaid.run({ nodes: diagrams });
})();
