import React from "react";
import { View, Text, Image, ScrollView } from "react-native";
import { NativeStackScreenProps } from "@react-navigation/native-stack";
import { styles } from "./Styles";

type RootStackParamList = {
  Camera: undefined;
  Preview: { photoUri: string };
  Report: {
    photoUri: string;
    classe: string;
    confianca: number;
    tratamento: string;
  };
};

type Props = NativeStackScreenProps<RootStackParamList, "Report">;

type Section = {
  title?: string;
  content: string;
  boldInContent?: { text: string; boldText: string }[];
};

export function Report({ route }: Props) {
  const { photoUri, classe, tratamento } = route.params;

  const capitalize = (s: string) =>
    s
      .split(" ")
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
      .join(" ");
  const condicao = classe.replace("Tomato___", "");
  const formattedCondicao = capitalize(condicao.replace(/_/g, " "));

  const sections: Section[] = [
    { title: "Tratamento:", content: tratamento },
    {
      title: "Magnesium Chlorosis",
      content:
        "Magnesium chlorosis is a nutritional deficiency that occurs when a plant has low levels of magnesium (Mg), an essential nutrient for the formation of chlorophyll. As a result, photosynthesis is reduced and the plant weakens.",
    },
    {
      title: "Main Symptoms",
      content:
        "• Yellowing between the veins of older leaves (interveinal chlorosis)\n• Veins remain green, creating a 'network' appearance\n• In severe cases, necrotic (dead) spots may appear in the yellowed areas\n• Reduced growth and premature leaf drop",
    },
    {
      title: "Common Causes",
      content:
        "• Sandy or nutrient-poor soils\n• Excess potassium, calcium, or aluminum in the soil, which compete with magnesium for absorption\n• Leaching in very wet or over-watered soils",
    },
    {
      title: "How to Treat",
      content:
        "Apply magnesium-rich fertilizers, such as magnesium sulfate (Epsom salt) or dolomite (in acidic soils, it also helps adjust the pH).",
      boldInContent: [
        { text: "Soil Amendments: ", boldText: "Soil Amendments: " },
      ],
    },
    {
      content:
        "Spraying a magnesium sulfate solution (1 to 2%), allowing rapid absorption by the leaves.",
      boldInContent: [
        { text: "Foliar Application: ", boldText: "Foliar Application: " },
      ],
    },
  ];

  return (
    <ScrollView style={styles.containerReport}>
      <View style={styles.containerData}>
        <View style={styles.main}>
          <Image source={{ uri: photoUri }} style={styles.previewImage} />
          <View style={styles.responseTitle}>
            <Text style={styles.condition}>{formattedCondicao}</Text>
            <Text style={styles.title}>Folha de tomate</Text>
          </View>
        </View>

        <View style={{ marginTop: 16 }}>
          {sections.map((section, index) => (
            <View key={index} style={{ marginBottom: 16 }}>
              {section.title && (
                <Text style={styles.sectionTitle}>{section.title}</Text>
              )}
              {section.boldInContent ? (
                <Text style={styles.sectionText}>
                  {section.boldInContent.map((b, i) => (
                    <Text key={i} style={styles.boldText}>
                      {b.boldText}
                    </Text>
                  ))}
                  {section.content.replace(
                    section.boldInContent.map((b) => b.text).join(""),
                    ""
                  )}
                </Text>
              ) : (
                <Text style={styles.sectionText}>{section.content}</Text>
              )}
            </View>
          ))}
        </View>
      </View>
    </ScrollView>
  );
}
