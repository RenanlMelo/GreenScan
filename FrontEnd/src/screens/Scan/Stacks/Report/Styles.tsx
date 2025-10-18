import { StyleSheet, Dimensions } from "react-native";

const { width, height } = Dimensions.get("window");

export const styles = StyleSheet.create({
  containerReport: {
    flex: 1,
    backgroundColor: "#edf2e1",
    padding: 12,
  },
  containerData: {
    justifyContent: "center",
    backgroundColor: "#ffffff",
    padding: 16,
    marginTop: 24,
    borderRadius: 20,
  },
  main: {
    backgroundColor: "#2B4336",
    padding: 12,
    width: width - 56,
    marginHorizontal: "auto",
    flexDirection: "row",
    borderRadius: 15,
  },
  previewImage: {
    width: 150,
    height: 150,
    borderRadius: 10,
  },
  responseTitle: {
    padding: 10,
    gap: 6,
    flex: 1,
  },
  title: {
    color: "#c4ceaeff",
  },
  condition: {
    color: "#fff",
    fontSize: 16,
    flexWrap: "wrap",
    flexShrink: 1,
  },
  sectionTitle: {
    marginBottom: 8,
    fontWeight: "bold",
  },
  sectionText: {
    marginBottom: 16,
    lineHeight: 20,
  },
  boldText: {
    fontWeight: "bold",
  },
});
