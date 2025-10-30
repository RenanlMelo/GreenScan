import { StyleSheet } from "react-native";

export const styles = StyleSheet.create({
  container: {
    padding: 16,
    backgroundColor: "#edf2e1",
    flex: 1,
  },

  title: {
    fontSize: 18,
    fontWeight: "700",
    marginBottom: 12,
    color: "#0F172A",
  },

  item: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "#FBFBFB",
    borderRadius: 12,
    padding: 12,
    marginBottom: 10,
    shadowColor: "#000",
    shadowOpacity: 0.05,
    shadowRadius: 12,
    elevation: 1,
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
    justifyContent: "flex-start",
    paddingTop: 4,
    height: "100%",
  },

  name: {
    fontSize: 16,
    fontWeight: "600",
    color: "#0B1220",
  },

  situation: {
    fontSize: 13,
    color: "#475569",
    marginTop: 4,
  },

  right: {
    alignItems: "flex-end",
    justifyContent: "space-between",
    height: 70,
    marginLeft: 8,
    flexDirection: "column",
  },
  deleteButton: {
    width: 40,
    height: 40,
    borderRadius: 9,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "rgba(255,59,48,0.08)", // soft red background
  },

  empty: {
    padding: 24,
    alignItems: "center",
  },

  emptyText: {
    color: "#64748B",
  },
  time: {
    fontSize: 13,
    color: "#999",
  },
});
