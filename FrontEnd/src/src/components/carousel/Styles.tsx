import { StyleSheet, Dimensions } from "react-native";
const { width } = Dimensions.get("window");

export const styles = StyleSheet.create({
  carousel: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
  },
  title: {
    marginBottom: 10,
  },
  card: {
    width: width - 16,
    height: 250,
    marginHorizontal: 8,
    borderRadius: 12,
    overflow: "hidden",
  },
  image: {
    width: "100%",
    height: "100%",
  },

  overlayContainer: {
    position: "absolute",
    bottom: 0,
    width: "100%",
    padding: 12,
    paddingBottom: 28,
    backgroundColor: "rgba(0,0,0,0.4)",
  },
  imageTitle: {
    color: "#fff",
    fontSize: 18,
    fontWeight: "bold",
    marginBottom: 4,
  },
  imageBody: {
    color: "#ddd",
    fontSize: 14,
  },

  dotsContainer: {
    width: "100%",
    flexDirection: "row",
    justifyContent: "center",
    alignItems: "center",
    marginTop: 8,
    position: "absolute",
    bottom: 8,
    left: 0,
  },
  dot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: "#909090",
    marginHorizontal: 4,
  },
  activeDot: {
    backgroundColor: "#fff",
  },
});
