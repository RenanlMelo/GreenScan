import { StyleSheet, Dimensions } from "react-native";
const { width } = Dimensions.get("window");

export const styles = StyleSheet.create({
  container: {
    flex: 1,
    width: width,
    height: "100%",
    paddingTop: 40,
    backgroundColor: "#edf2e1",
    gap: 24,
  },
});
