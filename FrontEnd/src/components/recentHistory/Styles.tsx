// Styles.js
import { StyleSheet, Dimensions } from "react-native";

const { width } = Dimensions.get("window");

export const styles = StyleSheet.create({
  container: {
    width: width - 16,
    backgroundColor: "#fff",
    borderRadius: 16,
    padding: 8,
    alignSelf: "center",
    marginTop: 16,
    marginBottom: 64,
    shadowColor: "#00000060",
    shadowOpacity: 0.05,
    shadowOffset: { width: 0, height: 2 },
    shadowRadius: 6,
    elevation: 3,
  },
  title: {
    fontSize: 18,
    fontWeight: "600",
    marginBottom: 12,
    color: "#1a1a1a",
  },
  item: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "#FBFBFB",
    borderRadius: 12,
    padding: 12,
    marginBottom: 10,
  },
  image: {
    width: 56,
    height: 56,
    borderRadius: 8,
    marginRight: 12,
    backgroundColor: "#eee",
  },
  info: {
    flex: 1,
  },
  name: {
    fontSize: 16,
    fontWeight: "600",
    color: "#2d2d2d",
  },
  situation: {
    fontSize: 14,
    color: "#6c757d",
    marginTop: 2,
  },
  time: {
    fontSize: 13,
    color: "#999",
  },
});
