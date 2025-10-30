import { StyleSheet, Dimensions } from "react-native";
const { width } = Dimensions.get("window");

export const styles = StyleSheet.create({
  container: {
    width: width - 16,
    borderRadius: 12,
    alignSelf: "center",
    backgroundColor: "white",
    marginTop: 16,
    paddingTop: 8,
    paddingBottom: 20,
    paddingHorizontal: 12,
    gap: 10,
  },
  textContainer: {
    flexDirection: "row",
    alignItems: "center",
    gap: 12,
    marginTop: 12,
    marginBottom: 4,
    fontWeight: "700",
  },
  scanText: {
    fontSize: 16,
    color: "#517861",
    flexShrink: 1,
  },
  icon: {
    borderRadius: 8,
    borderWidth: 2,
    borderColor: "#B2D5B8",
    padding: 8,
  },
  button: {
    padding: 12,
    backgroundColor: "#517861",
    width: "100%",
    borderRadius: 30,
  },
  buttonText: {
    color: "white",
    fontWeight: "700",
    paddingVertical: 4,
    fontSize: 18,
    textAlign: "center",
  },
});
