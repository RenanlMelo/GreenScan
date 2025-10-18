// Styles.js
import { StyleSheet, Dimensions } from "react-native";

const { width } = Dimensions.get("window");

export const styles = StyleSheet.create({
  container: {
    width: width - 16,
    backgroundColor: "#fff",
    borderRadius: 16,
    padding: 16,
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
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: "#f0f0f0",
  },
  image: {
    width: 44,
    height: 44,
    borderRadius: 12,
    backgroundColor: "#e6f0ea",
    marginRight: 12,
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
