import { Dimensions, StyleSheet } from "react-native";
const { width, height } = Dimensions.get("window");

export const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  camera: {
    flex: 1,
  },

  buttonContainer: {
    position: "absolute",
    bottom: 32,
    right: 32,
    flexDirection: "column",
    gap: 16,
    padding: 8,
    backgroundColor: "#303030aa",
    borderRadius: 24,
    alignSelf: "center",
    alignItems: "center",
  },
  button: {
    alignItems: "center",
    justifyContent: "center",
    paddingHorizontal: 12,
    paddingVertical: 8,
    minWidth: 44,
    borderRadius: 20,
  },
});
